# ConvSegFormer
ConvSegFormer - A convolution aided SegFormer architecture for detection of discontinuities in wrapped interferometric phase imagery of Sea Ice

![](/Images/ConvSegFormerB0.svg)

*ConvSegFormer architecture*

![](/Images/MESAPP.svg)

*Modified Efficient Self Attention (MESA)*

### Abstract

Transformers are more expressive than convolutions due to their global receptive field and lack of inherent biases. However, they require large amounts of training data to use this expressivity, which might hinder their application in scenarios with scarce training data. In the past, several works explored the idea of adding convolutions to transformer architecture to mitigate this issue. Yet, they underutilize hierarchical features extracted by these convolutions. We propose **ConvSegformer**, a hybrid deep learning model, by modifying the SegFormer architecture. We add a parallel convolutional encoder to extract hierarchical features that guide the SegFormer model. Furthermore, we replace the simple decoding scheme of the SegFormer architecture with a progressive upsampling method using features from both SegFormer and convolutional encoders. Finally, we modify the efficient self-attention module in the SegFormer branch to integrate transformer and convolution features. We demonstrate the efficacy of the proposed method in detecting discontinuities in Gamma Portable Radar Interferometer (GPRI) images.

### Training instructions

Download the dataset from above and run the following command to train the larger model on **2** gpus with a batch size of **6**.

```bash
python train.py 0.0003 61 ConvSegFormer 2 8
```

### Results

![](/Images/ResultsTable.png)

If you find our work useful, please use the following to cite it - 

```
@inproceedings{dulam2023convsegformer,
  title={ConvSegFormer-A Convolution Aided SegFormer Architecture for Detection of Discontinuities in Wrapped Interferometric Phase Imagery of Sea Ice},
  author={Dulam, Rohit Venkata Sai and Fedders, Emily R and Mahoney, Andrew R and Kambhamettu, Chandra},
  booktitle={Image Analysis: 23rd Scandinavian Conference, SCIA 2023, Sirkka, Finland, April 18--21, 2023, Proceedings, Part II},
  pages={203--213},
  year={2023},
  organization={Springer}
}
```
