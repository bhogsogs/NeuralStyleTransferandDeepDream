# Neural Style Transfer From Scratch with DeepDream
An implementation of the neural style transfer algorithm along with the DeepDream algorithm to incorporate features of style inside images.

## DeepDream Algorithm (Inceptionism)

DeepDream is an experiment that visualizes the patterns learned by a neural network. Similar to when a child watches clouds and tries to interpret random shapes, DeepDream over-interprets and enhances the patterns it sees in an image. It does so by forwarding an image through the network, then calculating the gradient of the image with respect to the activations of a particular layer. The image is then modified to increase these activations, enhancing the patterns seen by the network, and resulting in a dream-like image. This process was dubbed "Inceptionism" (a reference to InceptionNet, and the movie Inception).

This notebook presents an implementation of the DeepDream algorithm. We make use of particularly 3 layers of a pretrained InceptionV3 Convnet along with their corresponding coefficients(basically, a way to tune the influence of the chosen layers on the input image), and gradually implement changes in our input image so that the features encapsulated in these three chosen layers slowly become more promiment in our input image. The key part is implementing the gradient ascent algorithm, where  defined by the equation :-

$\text{Image}\_\{n+1\} = \text{Image}\_\{n\} + \alpha \nabla_{\text{Image}} f(\text{Image}_{n})$

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

Here, we try to implement the neural style transfer algorithm from scratch on the image that we obtained after subjecting it to the DeepDream algorithm. We leveraged a pretrained VGG19 model and picked up the intermediate layers of the same to quantify the content and style images. The key part was to define loss functions to calculate content, style and total loss; an MSE loss function for content loss, and a Gram Matrix based style loss for style differences:-
   
_Content loss:_

$L\_{content}(\vec{p}, \vec{x}, l) = \frac{1}{2} \sum\_{i,j} (F^l\_{ij} - P^l\_{ij})^2$

* $\text{Target Image} (\vec{p})$ is the reference image,
* $\text{Generated Image} (\vec{x})$ is the image produced by the algorithm
* $l$ is the index of the layer
* $F_{ij}^l$ is the feature representation of $\vec{x}$ at layer $l$
* $P_{ij}^l$ is the feature representation of $\vec{p}$ at layer $l$

_Style loss:_

$\mathcal{L}\_{\text {style }}(\vec{a}, \vec{x})=\sum\_l w\_l E\_l$

* $\text{Reference Style Image} (\vec{a})$ is the style reference image
*  $L$ is the total number of layers
*  $w_l$ is the weight for layer $l$
*  $E_l$ is the style loss at layer $l$
  
_Total loss:_

$L\_{total}(\vec{p}, \vec{a}, \vec{x}) = \alpha L\_{content}(\vec{p}, \vec{x}) + \beta L\_{style}(\vec{a}, \vec{x})$

* $\alpha$ and $\beta$ are hyperparameters that control the influence of content and style, respectively.
  
5. Perform gradient descent optimization to generate the styled image

## Results

After running the notebook on a local CPU after a compute time of roughly ~x hours, we obtained the final images that can be obtained in the "results" folder.

## Usage

Please refer to the Jupyter notebook for more information on how to use these custom classes and the specific implementation details.
