import struct

import torch
from transformers import BertForSequenceClassification

def serialize_fp32(file, tensor):
    """ writes one fp32 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)

def version1_export(model, filepath):
    version = 1
    
    
    out_file = open(filepath, 'wb')
    
    out_file.write(struct.pack('I', 0x616b3432))
    out_file.write(struct.pack('i', version))
    
    vocab_size = 30522
    max_position_embeddings = 512
    token_type_size = 2
    n_layers = 12
    hidden_dim = 768
    num_attention_heads = 12
    num_layers = 12
    
    header = struct.pack('iiiiiii', vocab_size, max_position_embeddings, token_type_size,
                         n_layers, hidden_dim, num_attention_heads, num_layers)
    
    out_file.write(header)
    
    pad = 256 - out_file.tell()
    assert pad >= 0
    out_file.write(b'\0' * pad)
    weights = [model.bert.embeddings.word_embeddings.weight,
               model.bert.embeddings.position_embeddings.weight,
               model.bert.embeddings.token_type_embeddings.weight,
               model.bert.embeddings.LayerNorm.weight,
               model.bert.embeddings.LayerNorm.bias,
               *[model.bert.encoder.layer[layer_index].attention.self.query.weight for layer_index, _ in enumerate(model.bert.encoder.layer)],
               *[model.bert.encoder.layer[layer_index].attention.self.query.bias for layer_index, _ in enumerate(model.bert.encoder.layer)],
               *[model.bert.encoder.layer[layer_index].attention.self.key.weight for layer_index, _ in enumerate(model.bert.encoder.layer)],
               *[model.bert.encoder.layer[layer_index].attention.self.key.bias for layer_index, _ in enumerate(model.bert.encoder.layer)],
               *[model.bert.encoder.layer[layer_index].attention.self.value.weight for layer_index, _ in enumerate(model.bert.encoder.layer)],
               *[model.bert.encoder.layer[layer_index].attention.self.value.bias for layer_index, _ in enumerate(model.bert.encoder.layer)],
               *[model.bert.encoder.layer[layer_index].attention.output.dense.weight for layer_index, _ in enumerate(model.bert.encoder.layer)],
               *[model.bert.encoder.layer[layer_index].attention.output.dense.bias for layer_index, _ in enumerate(model.bert.encoder.layer)],
               *[model.bert.encoder.layer[layer_index].attention.output.LayerNorm.weight for layer_index, _ in enumerate(model.bert.encoder.layer)],
               *[model.bert.encoder.layer[layer_index].attention.output.LayerNorm.bias for layer_index, _ in enumerate(model.bert.encoder.layer)],
               *[model.bert.encoder.layer[layer_index].intermediate.dense.weight for layer_index, _ in enumerate(model.bert.encoder.layer)],
               *[model.bert.encoder.layer[layer_index].intermediate.dense.bias for layer_index, _ in enumerate(model.bert.encoder.layer)],
               *[model.bert.encoder.layer[layer_index].output.dense.weight for layer_index, _ in enumerate(model.bert.encoder.layer)],
               *[model.bert.encoder.layer[layer_index].output.dense.bias for layer_index, _ in enumerate(model.bert.encoder.layer)],
               *[model.bert.encoder.layer[layer_index].output.LayerNorm.weight for layer_index, _ in enumerate(model.bert.encoder.layer)],
               *[model.bert.encoder.layer[layer_index].output.LayerNorm.bias for layer_index, _ in enumerate(model.bert.encoder.layer)],
               model.bert.pooler.dense.weight,
               model.bert.pooler.dense.bias,
               model.classifier.weight,
               model.classifier.bias]
    
    for w in weights:
        serialize_fp32(out_file, w)
        
    out_file.close()
    print(f"wrote {filepath}")
    

if __name__ == '__main__':
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    file_path = "pytorch_model.bin"
    version1_export(model, file_path)

    
    