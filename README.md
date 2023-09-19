# GAN Project : I’m Something of a Painter Myself
This project has been initiated to better understand the development and trianing of GAN Models

The two models will work against each other, with the generator trying to trick the discriminator, and the discriminator trying to accurately classify the real vs. generated images.

Style transfer is the technique of reconstructing images in the style of another image.

I have chosen to use Andy Warhol. Perosnally i like the use of his coolours, high contrast and deep saturation. Its somethingthat notonly sticks to art but can be a place in a home and make you think "that looks cool bruv".

## The Challenge:
A GAN consists of at least two neural networks: a generator model and a discriminator model. The generator is a neural network that creates the images. For our competition, you should generate images in the style of Monet. This generator is trained using a discriminator. The two models will work against each other, with the generator trying to trick the discriminator, and the discriminator trying to accurately classify the real vs. generated images.

## details/developemnt (TO WORK ON)

Creating a GAN model to generate images in the style of Andy Warhol or apply an Andy Warhol style to input images is an interesting project. Here are the general steps you can follow to achieve this:

**1. Data Collection and Preprocessing:**
   - Gather a dataset of Andy Warhol's artworks. You mentioned having around 130 images, which can be a good starting point.
   - Preprocess the images, ensuring they have consistent dimensions, color channels, and are scaled to a common size.

**2. Build a GAN Model:**
   - Create a generator and a discriminator network as described earlier for GANs in TensorFlow.
   - You may want to explore different architectures for the generator and discriminator to capture the style of Andy Warhol's artwork effectively.

**3. Data Augmentation:**
   - To improve the quality and diversity of generated images, consider using data augmentation techniques, such as rotation, scaling, and flipping, during training.

**4. Training:**
   - Train your GAN model on the Andy Warhol dataset. The generator should learn to generate images in the style of Andy Warhol, while the discriminator should learn to distinguish real Andy Warhol images from generated ones.
   - Experiment with hyperparameters like learning rate, batch size, and training duration to get good results.

**5. Style Transfer (Optional):**
   - Once you have a trained generator, you can use it for style transfer. To apply Andy Warhol's style to an input image, feed the input image through the generator to get a stylized output.

**6. Evaluation:**
   - Evaluate the quality of generated images using metrics like Inception Score or Frechet Inception Distance (FID).
   - Fine-tune your model and training process based on the evaluation results.

**7. Save and Deploy:**
   - Save the trained generator model for future use.
   - If you want to create an application or tool that allows users to apply the Andy Warhol style to their own images, you can deploy the model as part of a web application or mobile app.

**8. Experiment and Iterate:**
   - GAN training can be a bit trial and error. Don't hesitate to iterate on your model architecture, training process, and hyperparameters to achieve the desired results.

Remember that generating high-quality art in the style of a specific artist like Andy Warhol can be challenging, and the results may vary. It's essential to have realistic expectations and be prepared for some experimentation to fine-tune your model for the best results. Additionally, you can explore techniques like neural style transfer to further refine the style transfer aspect of your project.

## generator descirmnator
- small dataset wouldnt be suited for complex models to avoid over fitting
-best suited for style transfer problems. for image generation, a large dataset would be best suited.
-generator Encoder-Decoder with Pretrained VGG Encoder:
descrimnator: Style Evaluation Discriminator, The discriminator aims to distinguish between stylized images and non-stylized images.

## training
    Initialization:
        Initialize the generator and discriminator models.
        Define loss functions for both the generator and discriminator.
        Choose hyperparameters such as learning rates, batch size, and the balance between generator and discriminator updates.

    Training Loop:

        The training loop consists of multiple iterations (epochs).

        In each iteration, you'll process batches of training data.

        For each batch, you'll perform the following steps:

        Training the Discriminator:
            Generate a batch of real images from your dataset.
            Generate a batch of fake images using the generator.
            Compute the discriminator's loss on the real and fake images.
            Update the discriminator's weights to minimize its loss.

        Training the Generator:
            Generate a batch of fake images using the generator.
            Compute the generator's loss, which typically consists of two components: adversarial loss (fooling the discriminator) and style loss (matching the desired style).
            Update the generator's weights to minimize its loss.

        Monitoring and Logging:
            Periodically, you should log and visualize various metrics, such as generator loss, discriminator loss, and the quality of generated images.
            You can use tools like TensorBoard for visualization.

        Evaluation:
            After each epoch or a certain number of iterations, you can evaluate the quality of the generated images using perceptual metrics or visual inspection.
            Make adjustments to the models or hyperparameters as needed.

        Save Checkpoints:
            Save checkpoints of your generator and discriminator models during training. This allows you to resume training from a specific point or use the trained models for inference.

    Termination:
        Decide on a termination condition, such as a maximum number of epochs or achieving a desired level of image quality.
        End the training loop when the condition is met.

    Inference and Style Transfer:
        After training, you can use the trained generator for style transfer. Encode an input image into the latent space and decode it with the generator to get the stylized output.

    Fine-Tuning (Optional):
        If the results are not satisfactory, you can fine-tune the models, adjust hyperparameters, or collect more data.


## Featured packages/APIs used
- TensorFlow
- Keras
- Jupyter Notebook



### Resources used
- https://www.youtube.com/watch?v=Nrsy6vF7rSw
- https://www.youtube.com/watch?v=Mng57Tj18pc
- https://www.youtube.com/watch?v=tX-6CMNnT64

- https://github.com/junyanz/CycleGAN
https://github.com/omerlux/Something-of-a-Painter

- https://keras.io/examples/generative/cyclegan/

- https://www.tensorflow.org/tutorials/generative/dcgan
- https://www.tensorflow.org/tutorials/generative/cyclegan

### Dataset
 - https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time
 I have specifically used the "Andy Warhol" set of pictures.