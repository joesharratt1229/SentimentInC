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
    int max_input_seq;
    int hidden_size;
    int max_position_embeddings;
    int n_layers;
    int token_type_size;
    int num_classes;
} ModelConfig;


typedef struct {
    float* w_q;
    float* b_q;
    float* w_k;
    float* b_k;
    float* w_v;
    float* b_v;
    float* w_o1;
    float* b_o1;
    float* w_ln1;
    float* b_ln1;
    float* w_o2;
    float* b_o2;
    float* w_o3;
    float* b_o3;
    float* w_ln2;
    float* b_ln2;
} BertLayer;


typedef struct {
    float* tokenEmbeddingTable;   //(num_tokens, embed_dim)
    float* positionEmbeddingTable;
    float* maskEmbeddingTable;
    float* w_ln_e;
    float* b_ln_e;
    BertLayer* bert_l;
    int num_layers;
    float* w_output;
    float* b_output;
    float* w_cls;
    float* b_cls;
} ModelWeights;



typedef struct {
    ModelWeights weights;
    ModelConfig config;
} Transformer;


typedef struct {
    float* embed_cache;  // (embed_dim, seq_len)
    float* norm0_cache; // (embed_dim, seq_len)
    float* key_cache;    // (head_dim, n_heads, seq_len, layer)
    float* val_cache;    // (head_dim, n_heads, seq_len, layer)
    float* query_cache;
    float* attention_cache; // (embed_dim, seq_len, layer)
    float* norm1_cache; // (embed_dim, seq_len, layer)
    float* norm2_cache;  // (embed_dim, seq_len, layer)
    float* hidden_activ_cache; // (embed_dim * 4, seq_len, layer)
    float* outputs;
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

    float temp_ss;
    float temp_mean;

    for (int i; i < total_size; i++)
    {
        temp_mean += x[i];
        temp_ss += pow(x[i], 2);
    }

    float var = temp_ss/total_size - (temp_mean * temp_mean)/total_size;
    float mean = temp_mean/total_size;

    int i;
    #pragma omp parallel for private(i);

    for (i = 0 ; i < total_size/embed_size; i++)
    {
        for(int j = 0; j < embed_size; j ++)
        {
            xout[i] = (x[i]-mean)/sqrt(temp_ss + epsilon) * gamma[i] + beta[i];
        }
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



float* forward(Transformer* transformer,
               HiddenReps* x,
               float* input_tokens,
               int num_tokens)

{
    ModelWeights* model = &transformer->weights;
    ModelConfig* config = &transformer->config;

    int embed_dim = config->hidden_size;
    int input_mask = 1;

    for(int i = 0; i < num_tokens; i++)
    {
        embedToken(model, x->embed_cache, input_tokens[i], embed_dim, i, input_mask);
    }

    //layerNorm(x->norm0_cache, x->embed_cache, embed_dim, )

}



/// Reading the input file

/*
typedef struct {
    float* w_q;
    float* b_q;
    float* w_k;
    float* b_k;
    float* w_v;
    float* b_v;
    float* w_o1;
    float* b_o1;
    float* w_ln1;
    float* b_ln1;
    float* w_o2;
    float* b_o2;
    float* w_o3;
    float* b_o3;
    float* w_ln2;
    float* b_ln2;
} BertLayer;

*/

void mapWeights(ModelWeights* weights, ModelConfig* c, BertLayer* bert_layers, float* ptr)
{
    printf("Hello");
    weights->tokenEmbeddingTable = ptr;
    ptr += (c->vocab_size* c->hidden_size);

    weights->positionEmbeddingTable = ptr;
    ptr+= (c->max_position_embeddings * c->hidden_size);

    weights->maskEmbeddingTable = ptr;
    ptr += (c->token_type_size * c->hidden_size);

    weights->w_ln_e = ptr;
    ptr += c->hidden_size;
    weights->b_ln_e = ptr;
    ptr += c->hidden_size;

    for(int i=0; i< c->n_layers; i++)
    {
        BertLayer* temp = &bert_layers[i];

        temp->w_q = ptr;
        ptr += (c->hidden_size * c->hidden_size);
        temp->b_q = ptr;
        ptr += c->hidden_size;

        temp->w_k = ptr;
        ptr += (c->hidden_size * c->hidden_size);
        temp->b_k = ptr;
        ptr += c->hidden_size;

        temp->w_v = ptr;
        ptr += (c->hidden_size * c->hidden_size);
        temp->b_v = ptr;
        ptr += c->hidden_size;

        temp->w_o1 = ptr;
        ptr += (c->hidden_size * c->hidden_size);
        temp->b_o1 = ptr;
        ptr += c->hidden_size;

        temp->w_ln1 = ptr;
        ptr += c->hidden_size;
        temp->b_ln1 = ptr;
        ptr += c->hidden_size;

        temp->w_o2 = ptr;
        ptr += (c->hidden_size * c->hidden_size * 4);
        temp->b_o2 = ptr;
        ptr += (c->hidden_size * 4);

        temp->w_o3 = ptr;
        ptr += (c->hidden_size * c->hidden_size * 4);
        temp->b_o3 = ptr;
        ptr += (c->hidden_size);

        temp->w_ln2 = ptr;
        ptr += c->hidden_size;
        temp->b_ln2 = ptr;
        ptr += c->hidden_size;        
    }

    weights->bert_l = bert_layers;

    weights->w_output = ptr;
    ptr += (c->hidden_size * c->hidden_size);
    weights->b_output = ptr;
    ptr += c->hidden_size;

    weights->w_cls = ptr;
    ptr += (c->hidden_size * c->num_classes);
    weights->b_cls = ptr;

    printf("Weights mapped");
}


void readFile(char* filePath, float** data, ModelWeights* weights, ModelConfig* config, BertLayer* bertLayers)
{
    FILE* fptr;
    fptr = fopen(filePath, "rb");
    if(fread(config, sizeof(config), 1, fptr) != 1){ exit(EXIT_FAILURE);};
    if (fptr == NULL){ printf("Unable to open file now exiting\n"); exit(EXIT_FAILURE);}
    fseek(fptr, 0, SEEK_END);

    printf("%d\n", config->hidden_size);
    
    ssize_t file_size = ftell(fptr);
    fclose(fptr);

    printf("File size is: %zd\n", file_size);

    int fs = open(filePath, O_RDONLY);
    if(fs==-1){ fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE);};
    *data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fs, 0); 
    if(*data == MAP_FAILED){fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE);}

    //mapWeights(weights, config, bertLayers, data);
}


int main()
{
    char* fileName = "../pytorch_model.bin";
    float* data;
    ModelWeights* weights;
    ModelConfig* config;
    BertLayer* bertlayers;

    readFile(fileName, &data, weights, config, bertlayers);
}

// leave above^ until understand how the weigghts are stored