# text to a picture. that's it.
1. collect data, extract text if needed
2. take basic model, train it on general handwriting
3. lora 

* GAN is the best method based on the training data and the data to be LoRA'd
* Probably will have to do some preprocessing of the data, sizes vary a lot and some fonts i don't really want?

i'm stupid i forgot the text embeddings entirely

i'm stupid i forgot to use the style extractor entirely so the entire time the i was sending in unencoded vectors of the wrong size. well well well.

7/8 log
* i have completely bastardized the model - downsampled instead of upsample, adjusted a bunch of random stuff, etc etc
* may have ruined the architecture
* need to fix loss now

5/8 log (i'm cooked pick up here next time) (fixed):
* convsublayer dims aren't matching, specifically after conv_skip (the convolution expects 64 but is getting 128)
* also i'm not sure if the transpose (to accomodate for channels in conv_skip) has any effects in the future. maybe reverse it?

architecture based on [this paper](https://arxiv.org/pdf/2011.06704):
[mnb](src/mobilenetbased.png)
[dataset](https://fki.tic.heia-fr.ch/databases/download-the-iam-on-line-handwriting-database)

# i'm gonna have to poke around with the model and data to get them to play nice. 
right now i'm just rewriting the model in pytorch but the data is not in the same form as theirs, maybe 

basically i have to edit the data to fix the dataloader and preprocess the data a bit more to get the style vectors

ok so the tensors are differently shaped which makes torch.stack unhappy (e.g [231, 1] and [422, 1]). usually collate_fn is used to fix this (via padding or whatever). no. strokes should be like that though? think about this.

## data
probably not enough so i'll use general handwriting to learn the words first. might be better to make it all caps

take the art, mask it if it's not bw, make it all bw (text detection and etc?)

is there a non-ml method

data/lineStrokes-all.tar.gz - the stroke xml for the online dataset 
data/lineImages-all.tar.gz - the images for the offline dataset 
ascii-all.tar.gz - the text labels for the dataset extract these contents and put them in the ./data directory

## graves paper notes
input is a real-valued pair (x1, x2) that defines the pen offset from the previous input

gaussians are used to predict x1 and x2

outputs are consist of EOS probability e and set of means, SDs, correlations, and mixture weights

vector y is obtained from a bunch of LSTM operations


