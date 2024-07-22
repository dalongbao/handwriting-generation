# text to a picture. that's it.
1. collect data, extract text if needed
2. take basic model, train it on general handwriting
3. lora 

* GAN is the best method based on the training data and the data to be LoRA'd
* Probably will have to do some preprocessing of the data, sizes vary a lot and some fonts i don't really want?

## data
probably not enough so i'll use general handwriting to learn the words first. might be better to make it all caps

take the art, mask it if it's not bw, make it all bw (text detection and etc?)

is there a non-ml method

gan is best option - text embedding + vqvae? -> picture, 

## graves paper notes
input is a real-valued pair (x1, x2) that defines the pen offset from the previous input

gaussians are used to predict x1 and x2

outputs are consist of EOS probability e and set of means, SDs, correlations, and mixture weights

vector y is obtained from a bunch of LSTM operations


