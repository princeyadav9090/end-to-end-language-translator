# iske andar hmara architecture pda hua h

import torch

class Encoder(torch.nn.Module):

    def _init_(self,src_lang_vocab_size,word_embedding_dim):
        super(Encoder,self)._init_()
        self.first_embedding_layer = torch.nn.Embedding(num_embeddings=src_lang_vocab_size,embedding_dim=word_embedding_dim)
        self.second_lstm_layer = torch.nn.LSTM(input_size=word_embedding_dim,hidden_size=word_embedding_dim,batch_first=True)

    def forward(self,X_padded_mini_batch):
        first_embedding_layer_out = self.first_embedding_layer(X_padded_mini_batch)
        encoder_output, (final_encoder_output,final_cell_state) = self.second_lstm_layer(first_embedding_layer_out)

        return encoder_output, (final_encoder_output,final_cell_state)


class Decoder(torch.nn.Module):

    def _init_(self,dst_lang_vocab_size,word_embedding_dim):
        super(Decoder,self)._init_()

        self.first_embedding_layer = torch.nn.Embedding(num_embeddings=dst_lang_vocab_size,
                                                       embedding_dim=word_embedding_dim)
        self.second_lstm_layer = torch.nn.LSTM(input_size=word_embedding_dim,
                                               hidden_size=word_embedding_dim,
                                              batch_first=True)
        self.prediction_layer = torch.nn.Linear(in_features=word_embedding_dim,out_features=dst_lang_vocab_size)
        # self.prediction_layer_activation = torch.nn.Softmax(dim=2)

    def forward(self,Y_padded_input_mini_batch,final_encoder_output,final_cell_state):

        first_embedding_layer_out = self.first_embedding_layer(Y_padded_input_mini_batch)
        decoder_lstm_layer_out, (final_decoder_lstm_layer_out, final_cell_state) = self.second_lstm_layer(first_embedding_layer_out,
                                                                                                         (final_encoder_output,
                                                                                                          final_cell_state))
        prediction = self.prediction_layer(decoder_lstm_layer_out)
        
        return prediction, (final_decoder_lstm_layer_out, final_cell_state)
        

class Seq2SeqEncDec(torch.nn.Module):
    def _init_(self,src_lang_vocab_size, dst_lang_vocab_size, word_embedding_dim):
        super(Seq2SeqEncDec, self)._init_()

        self.encoder = Encoder(src_lang_vocab_size,word_embedding_dim)
        self.decoder = Decoder(dst_lang_vocab_size,word_embedding_dim)

    def forward(self,X_padded_mini_batch,Y_padded_input_mini_batch):
        encoder_output, (final_encoder_output,final_cell_state) = self.encoder(X_padded_mini_batch)
        y_hat_mini_batch = self.decoder(Y_padded_input_mini_batch, final_encoder_output,final_cell_state)

        return y_hat_mini_batch