#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>


/*
attention, linear, layer_norm, linear, activation, linear, layer_norm
*/

typedef struct {
    int vocab_size;
    int max_position_embeddings;
    int token_type_size;
    int n_layers;
    int hidden_size;
    int num_attention_heads;
    int num_layers;
} ModelConfig;


typedef struct {
    float* tokenEmbeddingTable;   //(num_tokens, embed_dim)
    float* positionEmbeddingTable;
    float* maskEmbeddingTable;
    float* w_layer_norm_0;
    float* b_layer_norm_0;
    float* w_q;
    float* b_q;
    float* w_k;
    float* b_k;
    float* w_v;
    float* b_v;
    float* w_o1;
    float* b_o1;
    float* w_layer_norm_1;
    float* b_layer_norm_1;
    float* w_o2;
    float* b_o2;
    float* w_o3;
    float* b_o3;
    float* w_layer_norm_2;
    float* b_layer_norm_2;   
    float* w_output;
    float* b_output;
    float* w_cls;
    float* b_cls;
} ModelWeights;



typedef struct {
    ModelWeights* weights;
    ModelConfig* config;
} Transformer;


typedef struct {
    float* embed_cache;  // (embed_dim, seq_len)
    float* norm0_cache; // (embed_dim, seq_len)
    float* key_cache;    // (embed_dim, seq_len)
    float* val_cache;    // (embed_dim, seq_len)
    float* query_cache;
    float* k;
    float* q;
    float* v;
    float* attention_cache; // (seq_len, seq_len)
    float* attn_output_cache;
    float* mlp_input;
    float* norm1_cache; // (embed_dim, seq_len, layer)
    float* out1;
    float* norm2_cache;  // (embed_dim, seq_len, layer)
    float* out2; // (embed_dim * 4, seq_len, layer)
    float* transformer_layer_cache;
    float* lastHidden;
    float* output;
} HiddenReps;


void matmul(float* x, float* w, int output_dim, int hidden_dim, float* xout)
{
    // W (output_dim, hidden_dim) @ x (hidden_dim,) -> xout (d,)
    //compiler directive to exectute for loop parallelised threads
    int i;
    #pragma omp parallel for private(i);
    for (i = 0; i < output_dim; i++)
    {
        float val = 0;
        for (int j = 0; j<hidden_dim; j++)
        {
            val += (x[j] * w[i * hidden_dim + j]);
        }

        xout[i] = val;
    }
}

void addBias(float* x, float* b, float dim)
{
    int i;
    #pragma omp parallel for private(i);

    for(i=0; i<dim; i++)
    {
        x[i] += b[i];
    }
}

void softmax(float* x, int dim)
{
    float sum = 0;

    for (int i =0; i< dim; i++)
    {
        x[i] = expf(x[i]);
        sum += x[i];
    }

    for (int i; i< dim; i++)
    {
        x[i]/=sum ;
    }
}


void layerNorm(float* xout, 
               float* x, 
               float* gamma, 
               float* beta, 
               int embed_size, 
               int total_size, 
               float epsilon)
{

    //x_out (seq_len, embed_dim). x(seq_len, embed_dim). gamma(embed_dim2, embed_dim). beta(embed_dim2)  

    float temp_ss = 0.0f;
    float temp_mean = 0.0f;

    for (int i; i < total_size; i++)
    {
        temp_mean += x[i];
        temp_ss += pow(x[i], 2);
    }

    float var = temp_ss/total_size - (temp_mean * temp_mean)/total_size;
    float mean = temp_mean/total_size;

    printf("%f\n", temp_ss);
    int i;

    for (i = 0 ; i < total_size/embed_size; i++)
    {
        xout[i] = (x[i]-mean)/sqrt(var + epsilon) * gamma[i] + beta[i];
    }
}


void GeLU(float* x, int total_size)
{
    int i;
    #pragma omp parallel for private(i);

    for (i = 0; i < total_size; i++)
    {
        x[i] = x[i] * 0.5 * (1 + erf(x[i]/sqrt(2)));
    }
}

void my_tanh(float* x, int total_size)
{
    int i;
    #pragma omp parallel for private(i);

    for (i = 0; i < total_size; i++)
    {
        float temp = x[i];
        x[i] = (expf(temp)-expf(-temp))/(expf(temp)+expf(-temp));
    }

}


void embedToken(ModelWeights* model,
                float* output_cache,
                int token, 
                int embed_dim, 
                int posNum, 
                int input_mask)
        {
    float* inputEmbedding = model->tokenEmbeddingTable + (embed_dim * token);
    float* positionEmbedding = model->positionEmbeddingTable + (embed_dim * posNum);
    float* maskEmbedding = model->maskEmbeddingTable + (embed_dim * input_mask);

    for (int i = 0; i < embed_dim; i++)
    {
        output_cache[i + posNum * embed_dim] = inputEmbedding[i] + positionEmbedding[i] + maskEmbedding[i];
    }
}


void run_transformer_block(ModelWeights* weights, 
                           ModelConfig* config,
                           HiddenReps* cache, 
                           float* input,
                           float* output,
                           int layer_n, 
                           int num_tokens, 
                           int embed_dim)
{
    int attn_heads = config->num_attention_heads;
    int head_dim = embed_dim/attn_heads;

    float* wq=weights->w_q + (layer_n*embed_dim*embed_dim);
    float* wv=weights->w_v + (layer_n*embed_dim*embed_dim);
    float* wk=weights->w_k + (layer_n*embed_dim*embed_dim);
    

    for (int i = 0; i < num_tokens; i++)
    {
        float* att_input= &input[i*embed_dim];

        matmul(&input[i*embed_dim], weights->w_q, embed_dim, embed_dim, &cache->query_cache[i*embed_dim]);
        matmul(&input[i*embed_dim], weights->w_v, embed_dim, embed_dim, &cache->val_cache[i*embed_dim]);
        matmul(&input[i*embed_dim], weights->w_k, embed_dim, embed_dim, &cache->key_cache[i*embed_dim]);

        addBias(&cache->query_cache[i*embed_dim], weights->b_q, embed_dim);
        addBias(&cache->key_cache[i*embed_dim], weights->b_k, embed_dim);
        addBias(&cache->val_cache[i*embed_dim], weights->b_v, embed_dim);
    }


    for (int n_head=0; n_head<attn_heads; n_head++)
    {
        for (int tok=0; tok<num_tokens; tok++)
        {
            cache->q[tok*head_dim] = cache->query_cache[tok*embed_dim +head_dim*n_head];
            cache->k[tok*head_dim] = cache->key_cache[tok*embed_dim +head_dim*n_head];
            cache->v[tok*head_dim] = cache->val_cache[tok*embed_dim +head_dim*n_head];
        }

        float* kptr = cache->k;
        float* attn_che = cache->attention_cache;

        for (int tok=0; tok<num_tokens; tok++)
        {
            float* kptr = &cache->k[tok*head_dim];
            // indexes instead of manual arithmetic
            matmul(kptr, cache->q, num_tokens, head_dim, &attn_che[tok*num_tokens]);
            softmax(&attn_che[tok*num_tokens], num_tokens);
            matmul(&attn_che[tok*num_tokens], cache->v, head_dim, num_tokens, &cache->attn_output_cache[tok*embed_dim + head_dim * n_head]);
        }
    }

    for(int tok=0; tok<num_tokens; tok++)
    {
        matmul(&cache->attn_output_cache[embed_dim*tok], weights->w_o1, embed_dim, embed_dim, &cache->mlp_input[embed_dim*tok]);
    }

    addBias(cache->mlp_input, weights->b_o1, embed_dim);

    layerNorm(cache->norm1_cache, cache->mlp_input, weights->w_layer_norm_1, weights->b_layer_norm_0, embed_dim, num_tokens*embed_dim, 1e-5);

    for(int token = 0; token<num_tokens; token++)
    {
        matmul(&cache->norm0_cache[embed_dim*token], weights->w_o2, embed_dim*4, embed_dim, &cache->out1[token*embed_dim*4]);
        addBias(&cache->out1[token*embed_dim*4], weights->b_o2, embed_dim*4);
        GeLU(&cache->out1[embed_dim*token*4], embed_dim*4);
        matmul(&cache->out1[token*embed_dim*4], weights->w_o3, embed_dim, 4*embed_dim, &cache->out2[token*embed_dim]);
        addBias(&cache->out2[token*embed_dim], weights->b_o2, embed_dim);
    }

    layerNorm(output, cache->out2, weights->w_layer_norm_2, weights->b_layer_norm_2, embed_dim, num_tokens*embed_dim, 1e-5);
}


float* forward(Transformer* transformer,
               HiddenReps* x,
               int* input_tokens,
               int num_tokens)

{
    ModelWeights* weights = transformer->weights;
    ModelConfig* config = transformer->config;

    int embed_dim = config->hidden_size;
    int input_mask = 1;

    for(int i = 0; i < num_tokens; i++)
    {
        embedToken(weights, x->embed_cache, input_tokens[i], embed_dim, i, input_mask);
    }

    layerNorm(x->norm0_cache, x->embed_cache, weights->w_layer_norm_0, weights->b_layer_norm_0, embed_dim, embed_dim*num_tokens, 1e-5);
    
    *x->transformer_layer_cache = *x->norm0_cache;

    for (int layer=0; layer<config->n_layers; layer++)
    {
        run_transformer_block(weights, 
                              config, 
                              x, 
                              &x->transformer_layer_cache[layer*num_tokens*embed_dim], 
                              &x->transformer_layer_cache[(layer+1)*embed_dim*num_tokens],
                              layer,
                              num_tokens,
                              embed_dim);
    }

    float* output_ptr = &x->transformer_layer_cache[config->num_layers*embed_dim*num_tokens];
    // only process cls token at very beginning.

    matmul(output_ptr, weights->w_output, embed_dim, embed_dim, x->lastHidden);
    addBias(x->lastHidden, weights->b_output, embed_dim);
    my_tanh(x->lastHidden, embed_dim);
    matmul(x->lastHidden, weights->w_cls, 2, embed_dim, x->output);
    softmax(x->output, 2);
} 


// based on weights size need a function allocate memory on heap for transformer weights


void mapWeights(ModelWeights* weights, float* ptr, ModelConfig* c)
{
    weights->tokenEmbeddingTable = ptr;
    ptr += (c->vocab_size * c->hidden_size);

    weights->positionEmbeddingTable = ptr;
    ptr+= (c->max_position_embeddings * c->hidden_size);

    weights->maskEmbeddingTable = ptr;
    ptr += (c->token_type_size * c->hidden_size);

    weights->w_layer_norm_0 = ptr;
    ptr += c->hidden_size;
    weights->b_layer_norm_0 = ptr;
    ptr += c->hidden_size;

    weights->w_q = ptr;
    ptr += (c->n_layers*c->hidden_size*c->hidden_size);

    weights->b_q = ptr;
    ptr += (c->n_layers*c->hidden_size);

    weights->w_k = ptr;
    ptr += (c->n_layers*c->hidden_size*c->hidden_size);

    weights->b_k = ptr;
    ptr += (c->n_layers*c->hidden_size);

    weights->w_v = ptr;
    ptr += (c->n_layers*c->hidden_size*c->hidden_size);

    weights->b_v = ptr;
    ptr += (c->n_layers*c->hidden_size);

    weights->w_o1 = ptr;
    ptr += (c->n_layers*c->hidden_size*c->hidden_size);

    weights->b_o1 = ptr;
    ptr += (c->n_layers*c->hidden_size);

    weights->w_layer_norm_1 = ptr;
    ptr += (c->n_layers*c->hidden_size);

    weights->b_layer_norm_1 = ptr;
    ptr += (c->n_layers*c->hidden_size);

    weights->w_o2 = ptr;
    ptr += (c->n_layers*c->hidden_size*c->hidden_size*4);

    weights->b_o2= ptr;
    ptr += (c->n_layers*c->hidden_size * 4);

    weights->w_o3 = ptr;
    ptr += (c->n_layers*c->hidden_size*c->hidden_size*4);

    weights->b_o3 = ptr;
    ptr += (c->n_layers*c->hidden_size);

    weights->w_layer_norm_2 = ptr;
    ptr += (c->n_layers*c->hidden_size);

    weights->b_layer_norm_2 = ptr;
    ptr += (c->n_layers*c->hidden_size);

    weights->w_output = ptr;
    ptr += (c->hidden_size * c->hidden_size);
    weights->b_output = ptr;
    ptr += c->hidden_size;

    
    weights->w_cls = ptr;
    ptr += (c->hidden_size * 2);
    weights->b_cls = ptr;

    printf("Weights mapped");
}


void allocate_cache_memory(HiddenReps* h, ModelConfig* c, int seq_len)
{
    int n_elements = c->hidden_size*seq_len;
    int head_dim = c->hidden_size/c->num_attention_heads;

    h->embed_cache = (float *)calloc(n_elements, sizeof(float));
    h->norm0_cache = (float *)calloc(n_elements, sizeof(float));
    h->key_cache = (float *)calloc(n_elements, sizeof(float));
    h->query_cache = (float *)calloc(n_elements, sizeof(float));
    h->val_cache = (float *)calloc(n_elements, sizeof(float));

    h->v = (float *)calloc(seq_len*head_dim, sizeof(float));
    h->q = (float *)calloc(seq_len*head_dim, sizeof(float));
    h->k = (float *)calloc(seq_len*head_dim, sizeof(float));

    h->attention_cache = (float *)calloc(seq_len*seq_len, sizeof(float));
    h->attn_output_cache = (float *)calloc(n_elements, sizeof(float));
    h->mlp_input = (float *)calloc(n_elements, sizeof(float));
    h->norm1_cache = (float *)calloc(n_elements, sizeof(float));

    h->out1 = (float *)calloc(n_elements*4, sizeof(float));
    h->out2 = (float *)calloc(n_elements*4, sizeof(float));

    h->norm2_cache = (float *)calloc(n_elements*4, sizeof(float));

    h->transformer_layer_cache = (float *)calloc(n_elements*c->n_layers, sizeof(float));
    h->lastHidden = (float *)calloc(n_elements, sizeof(float));
    h->output = (float *)calloc(2, sizeof(float));
}


void readFile(char* filePath, ModelConfig* config, ModelWeights** weights)
{
    FILE* fptr;
    fptr = fopen(filePath, "rb");
    if (fptr == NULL){ printf("Unable to open file now exiting\n"); exit(EXIT_FAILURE);}
    fseek(fptr, 0, SEEK_END);
    
    ssize_t file_size = ftell(fptr);
    fclose(fptr);

    printf("File size is: %zd\n", file_size);

    int fs = open(filePath, O_RDONLY);
    if(fs==-1){ fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE);};
    float* data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fs, 0); 
    if(data == MAP_FAILED){fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE);}

    float *mptr = data + 64;

    //ModelWeights* 
    *weights = (ModelWeights *)malloc(file_size);
    mapWeights(*weights, mptr, config);
}


//free memory and add logic of fails to allocate memroy


int main()
{
    char* fileName = "../pytorch_model.bin";
    ModelConfig config = {
        .vocab_size = 30522,
        .max_position_embeddings = 512,
        .token_type_size = 2,
        .n_layers = 12,
        .hidden_size = 768,
        .num_attention_heads = 12,
        .num_layers = 12
    };

    int inputs[] = {101, 1045, 2572, 2025, 3110, 2204, 102};

    ModelWeights* weights;
    HiddenReps* reps = (HiddenReps*)malloc(sizeof(HiddenReps));

    readFile(fileName, &config, &weights);
    allocate_cache_memory(reps, &config, 7);

    Transformer t ={
        .weights = weights,
        .config = &config
    };

    forward(&t, reps, inputs, 7);
}