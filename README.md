# BackPropagation-MNIST

S6 -> PART 1
Back-Propagation in neural networks explained!

<img width="1103" alt="Screenshot 2024-02-25 at 6 28 53 PM" src="https://github.com/ayushmathur71/BackPropagation-MNIST/assets/30623714/53689bca-a7ef-4b12-bc92-12699a583167">


Consider above FC layer of a neural network. With the inputs -> weights -> activation function -> outputs -> Errors (Outputs compared with targets).

Let's start by calculating how each node is derived from the network:
<img width="185" alt="Screenshot 2024-03-02 at 7 40 34 AM" src="https://github.com/ayushmathur71/BackPropagation-MNIST/assets/30623714/f1bb1e90-d5d7-4021-9c74-76e83f42a25a">

With all the nodes & E1, E2 (Error terms) defined, we next start with calculation of partial derivatives. What is the partial derivative of E_total with respect to w5?
Answer is -> _∂E_total/∂w5 = ∂E1/∂w5 = ∂E1/∂a_o1 * ∂a_o1/∂o1 * ∂o1/∂w5_ (Remember chain rule in derivates, learnt in school??)
Key thing to notice here is, ∂E2/∂w5=0, as E2 does not changes OR gets impacted in anyway by w5. Intuition-> Notice there is no path that leads to E2 (from w5).
Similarly we calculate further simplify the above partial derivative into NN node & weight terms & we arrive at-> 
_∂E1/∂a_o1 = (a_o1 - t1)			
∂a_o1/∂o1 = a_o1 * (1-a_o1)			
∂o1/∂w5 = a_h1_

Similarly ->
_∂E_total/∂w5 = (a_o1 - t1) * a_o1 * (1-a_o1) * a_h1			
∂E_total/∂w6 = (a_o1 - t1) * a_o1 * (1-a_o1) * a_h2			
∂E_total/∂w7 = (a_o2 - t2) * a_o2 * (1-a_o2) * a_h1			
∂E_total/∂w8 = (a_o2 - t2) * a_o2 * (1-a_o2) * a_h2_

Now let's back propagate to a_h1 & a_h2, how should the partial derivative of Errors look like with respect to a_h1, a_h2?
<img width="592" alt="Screenshot 2024-03-02 at 7 57 10 AM" src="https://github.com/ayushmathur71/BackPropagation-MNIST/assets/30623714/64d97cad-06b2-44db-b62d-f55cfc65d1c4">

Finally we reach last layer and calculate partial derivatives of E_total with respect to w1, w2, w3, w4
<img width="712" alt="Screenshot 2024-03-02 at 8 09 34 AM" src="https://github.com/ayushmathur71/BackPropagation-MNIST/assets/30623714/929cb147-62be-4d9d-ae04-afc48a42d8a2">

All the partial derivates are the step values that we need to subtract from our current weights & arrive at new weights for our next iteration. Total Loss is calculated at the end of each iteration.

Total Loss = E1+E2, Let's plot a graph for all iterations & see how loss varies after each step weights are adjusted->
<img width="611" alt="Screenshot 2024-03-02 at 8 21 49 AM" src="https://github.com/ayushmathur71/BackPropagation-MNIST/assets/30623714/fd5a4b08-d47c-4bdf-be5c-e6eb02d9879c">

S6 -> PART 2
We achieved 99.49% validation accuracy with a DNN of 13,808 parameters.
First we defined the transforms-
1. ToTensor
2. Normalization with Mean & Std values
3. Rotation - This is angle by which we want to rotate the train images, as we identified that the train dataset contains rotated images.

Next we set the seed for reproducibility of the weights in our neural network.
We then create the train & test dataloaders, these dataloader objects helps us to iterate through the image dataset. We have printed a couple of images to analyse our training image set.

Next we define the neural network:
Key points to notice here, for every 3*3 convolution layer, we have added Batch Normalisation, Dropout value & RELU activation function.
Batch normalisation helps to normalise the scale of all images in the batch
Dropout value prevents overfitting & scales up the predicted value in proportion to the dropout%
RELU - activation function is required to only pass on the positive values

Neural Netwrok Architecture:
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)        
        x = self.convblock8(x)
As you notice, we have a max pooling layer after 3 convolution blocks (to reduce channel size). 
At the end right before the last convolution we have introduced a GAP layer, this helps increase our receptive field & decrease the output channel size to 1
Last convolution block convert 'n' channels into 10 channels that are passed to a log_softmax function.

Priniting the model summary - 13,808 parameters, then defining the training & testing functions that include backpropagation, loss calculation for each batch and printing of progress bar with accuracy details for each EPOCH. We have set LR at 0.01 and with 20 EPOCHS we are able to achieve a maximum validation accuracy of 99.49%.
