# ConvSegFormer
ConvSegFormer - A convolution aided SegFormer architecture for detection of discontinuities in wrapped interferometric phase imagery of Sea Ice

![](/Images/ConvSegFormerB0.svg)

*ConvSegFormer architecture*

![](/Images/MESAPP.svg)

*Modified Efficient Self Attention (MESA)*

### Abstract

Transformers are more expressive than convolutions due to their global receptive field and lack of inherent biases. However, they require large amounts of training data to use this expressivity, which might hinder their application in scenarios with scarce training data. In the past, several works explored the idea of adding convolutions to transformer architecture to mitigate this issue. Yet, they underutilize hierarchical features extracted by these convolutions. We propose **ConvSegformer**, a hybrid deep learning model, by modifying the SegFormer architecture. We add a parallel convolutional encoder to extract hierarchical features that guide the SegFormer model. Furthermore, we replace the simple decoding scheme of the SegFormer architecture with a progressive upsampling method using features from both SegFormer and convolutional encoders. Finally, we modify the efficient self-attention module in the SegFormer branch to integrate transformer and convolution features. We demonstrate the efficacy of the proposed method in detecting discontinuities in Gamma Portable Radar Interferometer (GPRI) images.
