# 📋 ShaDocFormer

<b><a href='https://arxiv.org/abs/2309.06670'>ShaDocFormer: A Shadow-Attentive Threshold Detector With Cascaded Fusion Refiner for Document Shadow Removal.</a> </b>
<div>
<span class="author-block">
  <a href='https://github.com/kilito777'>Weiwen Chen</a><sup> 
</span>,
  <span class="author-block">
    Yingtie Lei</a><sup>
  </span>,
  <span class="author-block">
   <a href='https://shenghongluo.github.io/'> Shenghong Luo</a><sup>
  </span>,
  <span class="author-block">
    Ziyang Zhou</a><sup>
  </span>,
  <span class="author-block">
    Mingxian Li</a><sup>
  </span> and
  <span class="author-block">
    <a href="https://www.cis.um.edu.mo/~cmpun/" target="_blank">Chi-Man Pun</a><sup> 
  </span>
</div>


<b>University of Macau</b>


In <b>_International Joint Conference on Neural Networks 2024 (IJCNN 2024)_<b>


[Paper](https://arxiv.org/abs/2309.06670) 


<img src="./result/result.png"/>

# ⚙️ Usage
## Installation
```
git clone https://github.com/kilito777/ShaDocFormer.git
cd ShaDocFormer
pip install -r requirements.txt
```

## Training
You may download the dataset first, and then specify TRAIN_DIR, VAL_DIR and SAVE_DIR in the section TRAINING in `config.yml`.

For single GPU training:
```
python train.py
```
For multiple GPUs training:
```
accelerate config
accelerate launch train.py
```
If you have difficulties with the usage of `accelerate`, please refer to <a href="https://github.com/huggingface/accelerate">Accelerate</a>.

# 💗 Acknowledgements
This work was supported in part by the Science and Technology Development Fund, Macau SAR, under Grant 0087/2020/A2 and Grant 0141/2023/RIA2.

# 🛎 Citation
If you find our work helpful for your research, please cite:
```bib

```
