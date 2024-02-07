# Neural Style Transfer From Scratch with DeepDream
An implementation of the neural style transfer algorithm along with the DeepDream algorithm to incorporate features of style inside images.

## DeepDream Algorithm (Inceptionism)

DeepDream is an experiment that visualizes the patterns learned by a neural network. Similar to when a child watches clouds and tries to interpret random shapes, DeepDream over-interprets and enhances the patterns it sees in an image. It does so by forwarding an image through the network, then calculating the gradient of the image with respect to the activations of a particular layer. The image is then modified to increase these activations, enhancing the patterns seen by the network, and resulting in a dream-like image. This process was dubbed "Inceptionism" (a reference to InceptionNet, and the movie Inception).

This notebook presents an implementation of the DeepDream algorithm. We make use of particularly 3 layers of a pretrained InceptionV3 Convnet along with their corresponding coefficients(basically, a way to tune the influence of the chosen layers on the input image), and gradually implement changes in our input image so that the features encapsulated in these three chosen layers slowly become more promiment in our input image. The key part is implementing the gradient ascent algorithm, where  defined by the equation :-

**Neural Style Transfer Update Rule**

$\text{Image}_{n+1} = \text{Image}_{n} + \alpha \nabla_{\text{Image}} f(\text{Image}_{n})$

where:

* $\text{Image}_{n+1}$ is the updated image after the $n^{th}$ iteration,
    
* $\text{Image}_n$ is the current image at the $n^{th}$ iteration, 
    
* $\alpha$ is the step size or learning rate, and
    
* $\nabla_{\text{Image}} f(\text{Image}_{n})$ is the gradient of the loss function $f$ with respect to the image at iteration $n$.


This iterative process enhances and amplifies features in the image, creating visually interesting patterns characteristic of the DeepDream algorithm.

It was also important to account for the lossess obtained while passing our image through the ConvNet and the loss in pixel data that arose after passing it through various kernels. An update equation for the same was also included in the algorithm implementation.

## Neural Style Transfer

Neural style transfer is an optimization technique used to take two images—a content image and a style reference image (such as an artwork by a famous painter)—and blend them together so the output image looks like the content image, but “painted” in the style of the style reference image. This is implemented by optimizing the output image to match the content statistics of the content image and the style statistics of the style reference image. These statistics are extracted from the images using a convolutional network.

![image](https://github.com/bhogsogs/Neural-Style-Transfer-From-Scratch-with-DeepDream/assets/134948011/c9e69c21-3cab-4821-935f-4279f1f85764)

Here, we try to implement the neural style transfer algorithm from scratch on the image that we obtained after subjecting it to the DeepDream algorithm. We leveraged a pretrained VGG19 model and picked up the intermediate layers of the same to quantify the content and style images. and the key steps are:

1. Load and preprocess the content and style images
2. Define VGG19 model and select output layers corresponding to content and style representations
3. Define loss functions to calculate content, style and total loss; an MSE loss function for content loss, and a Gram Matrix based style loss for style differences:-
   
  _Content loss:
  $L_{content}(\vec{p}, \vec{x}, l) = \frac{1}{2} \sum_{i,j} (F^l_{ij} - P^l_{ij})^2$
  
  _Style loss: 
  $$
\mathcal{L}_{\text {style }}(\vec{a}, \vec{x})=\sum_{l=0}^L w_l E_l
$$
  
  _Total loss:
  $L_{total}(\vec{p}, \vec{a}, \vec{x}) = \alpha L_{content}(\vec{p}, \vec{x}) + \beta L_{style}(\vec{a}, \vec{x})$
    
 where:
 - \( \alpha \) and \( \beta \) are hyperparameters that control the influence of content and style, respectively.

  
5. Perform gradient descent optimization to generate the styled image
   
The core style transfer logic is in training_loop() which trains the image over multiple iterations to minimize a weighted combination of content and style losses. The Gram Matrix loss function is used to measure the 

The main parameters to tweak are:

content_path and style_path: Images to use
iterations: Number of optimization steps
a and b: Weights for content vs style loss
