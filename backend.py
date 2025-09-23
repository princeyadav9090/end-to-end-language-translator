from transformers import AutoTokenizer
import torch 
from model import Seq2SeqEncDec

src_sent_tokenizer = AutoTokenizer.from_pretrained("google-T5/T5-base")


# This is the code of inference

def generate_translation(eng_sentence): # this function will accept a string like : "My name is Arjun" , whichj will be given by the user from frontend which is a streamlet page
    
    tokenized_eng_sentence = src_sent_tokenizer.tokenize(eng_sentence) # Tokenize the seng sentence and return tokenized list => output : ["My", "name", "is", "Jas"]
    token_ids = src_sent_tokenizer.convert_tokens_to_ids(tokenized_eng_sentence) # converts the words to numeric ids => output : [4,11,48,39]
    token_ids_tensor = torch.tensor(token_ids) # Convert list to tensor -> similar to numpy array => output : tensor[4,11,48,39]
    token_ids_tensor = torch.unsqueeze(token_ids_tensor,0) # Increases the dimension of the array (converts to 2D tensor) => output : tensor[[4,11,48,39]]

    # gpu hoga to gpu pr chla jyga sb kch
    if torch.cuda.is_available():
        device = torch.device("cuda")
        token_ids_tensor = token_ids_tensor.to(device)

    encoder_outputs,(final_encoder_output,final_candidate_cell_state) = network.encoder(token_ids_tensor) # encoder_output is a matrix(4,128) jo ki hr word ka meaning store krta h, final_enocder_output(long term memory), final_candidate_cell_state(short term memory) are vectors of 128 dimension which will go in decoder and they contain complete sentence meaning
    decoder_first_time_step_input = torch.tensor([[1]])

    if torch.cuda.is_available():
        encoder_outputs = encoder_outputs.to(device)
        final_encoder_output = final_encoder_output.to(device)
        final_candidate_cell_state = final_candidate_cell_state.to(device)
        decoder_first_time_step_input = decoder_first_time_step_input.to(device)

    decoder_first_time_step_output, (hidden_decoder_output, hidden_decoder_cell_state) = network.decoder(decoder_first_time_step_input,
                                                                                                         final_encoder_output,
                                                                                                         final_candidate_cell_state,) # hindi ka first word yha se niklega
    
    generated_token_id = torch.argmax(F.softmax(decoder_first_time_step_output[:,0,:],dim=1),1)
    generated_token_id = torch.unsqueeze(generated_token_id,1)
    
    hindi_translated_sentence = str()
    hindi_translated_sentence += " " + hindi_idx2vocab[generated_token_id.item()]
    if torch.cuda.is_available():
        generated_token_id = generated_token_id.to(device)
        hidden_decoder_output = hidden_decoder_output.to(device)
        hidden_decoder_cell_state = hidden_decoder_cell_state.to(device)
    
    for i in range(Nd-1): # Nd is the number of words in the longest hindi sentence ,ye ek demerit bhi h ki hm maximum itna bda hi sentence bna skte h, this thing in NLP is called context window

        # it is a vector of 7072 dimension , isis se hmara tranlation ayga
        decoder_first_time_step_output, (hidden_decoder_output, hidden_decoder_cell_state) = network.decoder(generated_token_id,
                                                                                                             hidden_decoder_output,
                                                                                                             hidden_decoder_cell_state) # hindi ka first word yha se niklega
    
        generated_token_id = torch.argmax(F.softmax(decoder_first_time_step_output[:,0,:],dim=1),1)
        generated_token_id = torch.unsqueeze(generated_token_id,1)
    
        if torch.cuda.is_available():
            generated_token_id = generated_token_id.to(device)
            hidden_decoder_output = hidden_decoder_output.to(device)
            hidden_decoder_cell_state = hidden_decoder_cell_state.to(device)
    
        if generated_token_id.item() == Vd["<EOS>"]:
            break

        hindi_translated_sentence += " " + hindi_idx2vocab[generated_token_id.item()]

    return hindi_translated_sentence