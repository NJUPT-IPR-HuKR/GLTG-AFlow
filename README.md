# Global-and Local-channel Self-attention Transformer and Attention Flow for Low-light Enhancement and Zero-element Pixels Restoration


## Introduction
GLTG-AFlow is a novel flow-based generative method for low-light image enhancement, noise suppression, and lost information restoration of zero-element pixels.
which consists of a global-and local-channel self-attention Transformer based conditional generator (GLTG) and an attention flow (AFlow). **Experiments show that GLTG-AFlow outperforms existing SOTA methods on benchmark low-light datasets, and low-light images with massive zero-element pixels.**

### Evaluation Metrics

<link rel="stylesheet" type="text/css" href="styles.css">


<table style="text-align: center; border-collapse: collapse; width: 100%;">
  <caption style="font-weight: bold; text-align: center;">
    Table 1. Metrics comparison of different methods on LOL-v2-real, LOL-v2-synthetic, SMID, SDSD-indoor, SDSD-outdoor, and SID datasets.  <strong>S</strong> denotes supervised learning-based method, and <strong>U</strong> denotes unsupervised learning-based method.  PSNR/SSIM/LPIPS values are obtained by re-training and testing codes of compared methods on our training and test datasets.  The best values are highlighted in bold.
  </caption>
  <thead>
      <tr>
      <th rowspan="2">Methods</th>
      <th rowspan="2">Type</th>
      <th colspan="3">LOLv2-real</th>
      <th colspan="3">LOLv2-synthetic</th>
      <th colspan="3">SMID</th>
      <th colspan="3">SDSD-indoor</th>
      <th colspan="3">SDSD-outdoor</th>
      <th colspan="3">SID</th>
    </tr>
    <tr>
      <th>PSNR&uarr;</th>
      <th>SSIM&uarr;</th>
      <th>LPIPS&darr;</th>
      <th>PSNR&uarr;</th>
      <th>SSIM&uarr;</th>
      <th>LPIPS&darr;</th>
      <th>PSNR&uarr;</th>
      <th>SSIM&uarr;</th>
      <th>LPIPS&darr;</th>
      <th>PSNR&uarr;</th>
      <th>SSIM&uarr;</th>
      <th>LPIPS&darr;</th>
      <th>PSNR&uarr;</th>
      <th>SSIM&uarr;</th>
      <th>LPIPS&darr;</th>
      <th>PSNR&uarr;</th>
      <th>SSIM&uarr;</th>
      <th>LPIPS&darr;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>RUAS</td><td>U</td><td>15.33</td><td>0.493</td><td>0.310</td><td>13.77</td><td>0.634</td><td>0.305</td><td>18.65</td><td>0.639</td><td>0.385</td><td>16.27</td><td>0.651</td><td>0.345</td><td>20.85</td><td>0.689</td><td>0.234</td><td>12.07</td><td>0.153</td><td>0.890</td>
    </tr>
    <tr>
      <td>SCI</td><td>U</td><td>17.30</td><td>0.540</td><td>0.308</td><td>15.43</td><td>0.744</td><td>0.233</td><td>19.74</td><td>0.653</td><td>0.359</td><td>21.19</td><td>0.723</td><td>0.328</td><td>18.09</td><td>0.538</td><td>0.296</td><td>12.84</td><td>0.150</td><td>0.931</td>
    </tr>
    <tr>
      <td>ZeroDCE++</td><td>U</td><td>18.43</td><td>0.587</td><td>0.306</td><td>17.69</td><td>0.815</td><td>0.186</td><td>17.06</td><td>0.612</td><td>0.405</td><td>20.53</td><td>0.671</td><td>0.360</td><td>17.01</td><td>0.517</td><td>0.328</td><td>13.07</td><td>0.146</td><td>0.905</td>
    </tr>
    <tr>
      <td>ZERO-IG</td><td>U</td><td>18.60</td><td>0.751</td><td>0.388</td><td>17.12</td><td>0.769</td><td>0.215</td><td>16.22</td><td>0.625</td><td>0.475</td><td>24.20</td><td>0.815</td><td>0.244</td><td>10.33</td><td>0.423</td><td>0.533</td><td>13.97</td><td>0.596</td><td>0.346</td>
    </tr>
    <tr>
      <td>EnlightenGAN</td><td>U</td><td>18.64</td><td>0.677</td><td>0.309</td><td>16.57</td><td>0.772</td><td>0.212</td><td>14.51</td><td>0.556</td><td>0.500</td><td>21.88</td><td>0.660</td><td>0.353</td><td>15.41</td><td>0.556</td><td>0.305</td><td>14.04</td><td>0.213</td><td>0.862</td>
    </tr>
    <tr>
      <td>PairLIE</td><td>U</td><td>19.89</td><td>0.773</td><td>0.234</td><td>19.07</td><td>0.794</td><td>0.230</td><td>13.36</td><td>0.532</td><td>0.507</td><td>19.46</td><td>0.689</td><td>0.284</td><td>12.37</td><td>0.434</td><td>0.368</td><td>14.55</td><td>0.272</td><td>0.831</td>
    </tr>
    <tr>
      <td>QuadPriors</td><td>U</td><td>20.47</td><td>0.808</td><td>0.198</td><td>16.10</td><td>0.752</td><td>0.251</td><td>15.68</td><td>0.600</td><td>0.433</td><td>22.22</td><td>0.777</td><td>0.206</td><td>18.29</td><td>0.661</td><td>0.209</td><td>15.79</td><td>0.500</td><td>0.521</td>
    </tr>
    <tr>
      <td>RetinexNet</td><td>S</td><td>15.66</td><td>0.661</td><td>0.589</td><td>19.24</td><td>0.798</td><td>0.246</td><td>14.91</td><td>0.539</td><td>0.531</td><td>26.48</td><td>0.824</td><td>0.218</td><td>14.47</td><td>0.557</td><td>0.340</td><td>13.69</td><td>0.198</td><td>0.828</td>
    </tr>
    <tr>
      <td>DeepUPE</td><td>S</td><td>17.61</td><td>0.536</td><td>0.353</td><td>19.70</td><td>0.841</td><td>0.155</td><td>22.31</td><td>0.685</td><td>0.282</td><td>21.15</td><td>0.669</td><td>0.361</td><td>25.74</td><td>0.782</td><td>0.185</td><td>15.66</td><td>0.198</td><td>0.896</td>
    </tr>
    <tr>
      <td>LLFlow</td><td>S</td><td>19.67</td><td>0.852</td><td>0.157</td><td>22.38</td><td>0.910</td><td>0.066</td><td>28.12</td><td>0.813</td><td>0.181</td><td>25.46</td><td>0.896</td><td>0.139</td><td>28.82</td><td>0.869</td><td>0.142</td><td>19.39</td><td>0.615</td><td>0.386</td>
    </tr>
    <tr>
      <td>MBPNet</td><td>S</td><td>19.95</td><td>0.837</td><td>0.144</td><td>24.55</td><td>0.918</td><td>0.066</td><td>26.93</td><td>0.771</td><td>0.178</td><td>28.82</td><td>0.882</td><td>0.126</td><td>22.87</td><td>0.754</td><td>0.186</td><td>18.75</td><td>0.441</td><td>0.596</td>
    </tr>
    <tr>
      <td>LLFormer</td><td>S</td><td>20.99</td><td>0.801</td><td>0.219</td><td>23.74</td><td>0.902</td><td>0.086</td><td>27.92</td><td>0.785</td><td>0.183</td><td>29.65</td><td>0.874</td><td>0.152</td><td>28.73</td><td>0.838</td><td>0.129</td><td>21.26</td><td>0.575</td><td>0.481</td>
    </tr>
    <tr>
      <td>MIRNet</td><td>S</td><td>21.18</td><td>0.840</td><td>0.145</td><td>25.08</td><td>0.920</td><td>0.070</td><td>28.67</td><td>0.810</td><td>0.180</td><td>27.83</td><td>0.882</td><td>0.138</td><td>29.17</td><td>0.871</td><td>0.152</td><td>20.87</td><td>0.605</td><td>0.460</td>
    </tr>
    <tr>
      <td>MIRNet-v2</td><td>S</td><td>21.37</td><td>0.833</td><td>0.153</td><td>24.29</td><td>0.923</td><td>0.064</td><td>28.49</td><td>0.804</td><td>0.174</td><td>28.74</td><td>0.890</td><td>0.122</td><td>30.07</td><td>0.866</td><td>0.152</td><td>21.80</td><td>0.630</td><td>0.405</td>
    </tr>
    <tr>
      <td>SNR</td><td>S</td><td>21.48</td><td>0.849</td><td>0.157</td><td>24.14</td><td>0.928</td><td>0.056</td><td>28.49</td><td>0.805</td><td>0.178</td><td>29.44</td><td>0.894</td><td>0.129</td><td>28.66</td><td>0.866</td><td>0.140</td><td>22.87</td><td>0.619</td><td>0.359</td>
    </tr>
    <tr>
      <td>Retinexformer</td><td>S</td><td>22.80</td><td>0.840</td><td>0.171</td><td>25.67</td><td>0.930</td><td>0.059</td><td><strong>29.15</strong></td><td><strong>0.815</strong></td><td>0.167</td><td>29.77</td><td>0.896</td><td>0.118</td><td>29.84</td><td>0.877</td><td>0.178</td><td><strong>24.44</strong></td><td><strong>0.680</strong></td><td><strong>0.344</strong></td>
    </tr>
    <tr>
      <td>SMG</td><td>S</td><td>24.03</td><td>0.820</td><td>0.169</td><td>24.98</td><td>0.894</td><td>0.092</td><td>26.97</td><td>0.725</td><td>0.211</td><td>26.89</td><td>0.802</td><td>0.166</td><td>26.33</td><td>0.809</td><td><strong>0.093</strong></td><td>22.63</td><td>0.541</td><td>0.377</td>
    </tr>
    <tr>
      <td><strong>GLTG-AFlow</strong></td><td>S</td><td><strong>25.71</strong></td><td><strong>0.894</strong></td><td><strong>0.103</strong></td><td><strong>26.80</strong></td><td><strong>0.951</strong></td><td><strong>0.037</strong></td><td>28.75</td><td>0.811</td><td><strong>0.164</strong></td><td><strong>31.47</strong></td><td><strong>0.913</strong></td><td><strong>0.102</strong></td><td><strong>30.12</strong></td><td><strong>0.883</strong></td><td>0.168</td><td>22.38</td><td>0.672</td><td>0.372</td>
    </tr>
  </tbody>
</table>

<table style="text-align: center; border-collapse: collapse; width: 100%;">
  <caption style="font-weight: bold; text-align: center;">
    Table 2. Metrics comparison of different methods on 5 datasets, where low-light images are with massive zero-element pixels those are obtained by 
combining with our zero-map set.
  </caption>
    <tr>
        <th rowspan="2" class="top-bordered right-bordered">Methods</th>
        <th colspan="3" class="top-bordered right-bordered">LOL-v2-real</th>
        <th colspan="3" class="top-bordered right-bordered">LOL-v2-synthetic</th>
        <th colspan="3" class="top-bordered right-bordered">SDSD-indoor</th>
        <th colspan="3" class="top-bordered right-bordered">SDSD-outdoor</th>
        <th colspan="3" class="top-bordered right-bordered">SID</th>
    </tr>
    <tr>
        <th>PSNR&uarr;</th>
        <th>SSIM&uarr;</th>
        <th class="right-bordered">LPIPS&darr;</th>
        <th>PSNR&uarr;</th>
        <th>SSIM&uarr;</th>
        <th class="right-bordered">LPIPS&darr;</th>
        <th>PSNR&uarr;</th>
        <th>SSIM&uarr;</th>
        <th class="right-bordered">LPIPS&darr;</th>
        <th>PSNR&uarr;</th>
        <th>SSIM&uarr;</th>
        <th class="right-bordered">LPIPS&darr;</th>
        <th>PSNR&uarr;</th>
        <th>SSIM&uarr;</th>
        <th class="right-bordered">LPIPS&darr;</th>
    </tr>
    <tr>
        <td class="right-bordered">HWMNet</td>
        <td>18.74</td>
        <td>0.719</td>
        <td class="right-bordered">0.639</td>
        <td>22.19</td>
        <td>0.834</td>
        <td class="right-bordered">0.278</td>
        <td>26.76</td>
        <td>0.856</td>
        <td class="right-bordered">0.191</td>
        <td>25.06</td>
        <td>0.805</td>
        <td class="right-bordered">0.214</td>
        <td>20.86</td>
        <td>0.589</td>
        <td class="right-bordered">0.495</td>
    </tr>
    <tr>
        <td class="right-bordered">LLformer</td>
        <td>18.95</td>
        <td>0.697</td>
        <td class="right-bordered">0.415</td>
        <td>22.67</td>
        <td>0.814</td>
        <td class="right-bordered">0.200</td>
        <td>28.05</td>
        <td>0.837</td>
        <td class="right-bordered">0.243</td>
        <td>28.72</td>
        <td>0.849</td>
        <td class="right-bordered">0.167</td>
        <td>20.91</td>
        <td>0.582</td>
        <td class="right-bordered">0.456</td>
    </tr>
    <tr>
        <td class="right-bordered">LLFlow</td>
        <td>19.19</td>
        <td>0.823</td>
        <td class="right-bordered">0.197</td>
        <td>22.19</td>
        <td>0.901</td>
        <td class="right-bordered">0.096</td>
        <td>27.45</td>
        <td>0.899</td>
        <td class="right-bordered">0.183</td>
        <td>28.90</td>
        <td>0.869</td>
        <td class="right-bordered">0.211</td>
        <td>18.63</td>
        <td>0.609</td>
        <td class="right-bordered">0.526</td>
    </tr>
    <tr>
        <td class="right-bordered">MIRNet</td>
        <td>21.17</td>
        <td>0.757</td>
        <td class="right-bordered">0.439</td>
        <td>22.13</td>
        <td>0.862</td>
        <td class="right-bordered">0.143</td>
        <td>28.11</td>
        <td>0.847</td>
        <td class="right-bordered">0.209</td>
        <td>28.83</td>
        <td>0.858</td>
        <td class="right-bordered"><b>0.163</b></td>
        <td>20.82</td>
        <td>0.605</td>
        <td class="right-bordered">0.439</td>
    </tr>
    <tr>
        <td class="right-bordered">Retinexformer</td>
        <td>21.29</td>
        <td>0.802</td>
        <td class="right-bordered">0.269</td>
        <td>24.73</td>
        <td>0.901</td>
        <td class="right-bordered">0.130</td>
        <td>30.08</td>
        <td>0.893</td>
        <td class="right-bordered">0.151</td>
        <td>28.85</td>
        <td>0.853</td>
        <td class="right-bordered">0.167</td>
        <td><b>21.75</b></td>
        <td>0.618</td>
        <td class="right-bordered">0.409</td>
    </tr>
    <tr>
        <td class="right-bordered">Zero-IG</td>
        <td>10.47</td>
        <td>0.155</td>
        <td class="right-bordered">0.913</td>
        <td>10.24</td>
        <td>0.290</td>
        <td class="right-bordered">0.815</td>
        <td>12.98</td>
        <td>0.270</td>
        <td class="right-bordered">0.819</td>
        <td>8.76</td>
        <td>0.114</td>
        <td class="right-bordered">0.809</td>
        <td>9.28</td>
        <td>0.063</td>
        <td class="right-bordered">0.933</td>
    </tr>
    <tr>
        <td class="right-bordered">PairLIE</td>
        <td>16.95</td>
        <td>0.446</td>
        <td class="right-bordered">0.683</td>
        <td>13.92</td>
        <td>0.371</td>
        <td class="right-bordered">0.749</td>
        <td>13.99</td>
        <td>0.344</td>
        <td class="right-bordered">0.750</td>
        <td>12.14</td>
        <td>0.275</td>
        <td class="right-bordered">0.686</td>
        <td>13.36</td>
        <td>0.201</td>
        <td class="right-bordered">0.867</td>
    </tr>
    <tr>
        <td class="right-bordered">Restormer</td>
        <td>21.70</td>
        <td>0.794</td>
        <td class="right-bordered">0.215</td>
        <td>24.02</td>
        <td>0.902</td>
        <td class="right-bordered">0.116</td>
        <td>29.15</td>
        <td>0.869</td>
        <td class="right-bordered">0.190</td>
        <td>27.56</td>
        <td>0.835</td>
        <td class="right-bordered">0.223</td>
        <td>21.16</td>
        <td>0.637</td>
        <td class="right-bordered">0.547</td>
    </tr>
    <tr class="bottom-bordered bold-top-border">
        <td class="right-bordered"><b>GLTG-AFlow</b></td>
        <td><b>24.17</b></td>
        <td><b>0.861</b></td>
        <td class="right-bordered"><b>0.155</b></td>
        <td><b>25.55</b></td>
        <td><b>0.930</b></td>
        <td class="right-bordered"><b>0.045</b></td>
        <td><b>30.49</b></td>
        <td><b>0.910</b></td>
        <td class="right-bordered"><b>0.112</b></td>
        <td><b>30.29</b></td>
        <td><b>0.880</b></td>
        <td class="right-bordered">0.172</td>
        <td>21.30</td>
        <td><b>0.643</b></td>
        <td class="right-bordered"><b>0.392</b></td>
    </tr>
</table>

(In the above two tables we remove the GT correction operation when obtaining the metrics of LLFlow for fair comparison. The enhanced image of SMG is with the size 512×512×3 that is different from the original ground truth (GT) image of each testing set, we rescale the enhanced images of SMG to have the same size with the original GT image of each testing set for fair comparison.)

<table style="text-align: center; border-collapse: collapse; width: 100%;">
    <caption style="font-weight: bold; text-align: center;">
Table 3. Metrics comparsion of NFL methods under the same training and testing tricks.
</caption>
    <tr> 
        <th rowspan="2" class="top-bordered right-bordered">Methods</th>
        <th colspan="2" class="top-bordered right-bordered">LOLv2-real</th>
        <th colspan="2" class="top-bordered right-bordered">LOLv2-synthetic</th>
        <th colspan="2" class="top-bordered right-bordered">SDSD-indoor</th>
        <th colspan="2" class="top-bordered">SDSD-outdoor</th>
    </tr>
    <tr>
        <th>PSNR&uarr;</th>
        <th class="right-bordered">SSIM&uarr;</th>
        <th>PSNR&uarr;</th>
        <th class="right-bordered">SSIM&uarr;</th>
        <th>PSNR&uarr;</th>
        <th class="right-bordered">SSIM&uarr;</th>
        <th>PSNR&uarr;</th>
        <th>SSIM&uarr;</th>
    </tr>
    <tr>
        <td class="right-bordered">LLFlow</td>
        <td>26.53</td>
        <td class="right-bordered">0.892</td>
        <td>26.23</td>
        <td class="right-bordered">0.943</td>
        <td>-</td>
        <td class="right-bordered">-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td class="right-bordered">LL-SKF</td>
        <td>28.45</td>
        <td class="right-bordered">0.900</td>
        <td>29.11</td>
        <td class="right-bordered">0.953</td>
        <td>-</td>
        <td class="right-bordered">-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td class="right-bordered">GLARE</td>
        <td>28.98</td>
        <td class="right-bordered">0.905</td>
        <td>29.84</td>
        <td class="right-bordered">0.958</td>
        <td>30.10</td>
        <td class="right-bordered">0.896</td>
        <td>30.85</td>
        <td>0.884</td>
    </tr>
    <tr class="bottom-bordered bold-top-border">
        <td class="right-bordered"><b>GLTG-AFlow</b></td>
        <td><b>30.08</b></td>
        <td class="right-bordered"><b>0.909</b></td>
        <td><b>30.19</b></td>
        <td class="right-bordered"><b>0.960</b></td>
        <td><b>31.67</b></td>
        <td class="right-bordered"><b>0.912</b></td>
        <td><b>31.95</b></td>
        <td><b>0.904</b></td>
    </tr>
</table>

## Dataset

- LOLv2 (Real & Synthetic): Please refer to the papaer [[From Fidelity to Perceptual Quality: A Semi-Supervised Approach for Low-Light Image Enhancement (CVPR 2020)]](https://github.com/flyywh/CVPR-2020-Semi-Low-Light).

- SID & SMID & SDSD (indoor & outdoor): Please refer to the paper [[SNR-aware Low-Light Image Enhancement (CVPR 2022)]](https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance).




## Testing

### Pre-trained Models

Evaluation Metrics of GLTG-AFlow in Table 1 are obtained by pre-trained models via the following links

LOL-v2-real:[[Google Drive]](https://drive.google.com/file/d/1WIYhozFftZxjx1rx3LIN4Z_1QFE_JlER/view?usp=sharing).
LOL-v2-syn:[[Google Drive]](https://drive.google.com/file/d/1P9xkb_nJYL9W9IsY0vw8-seCIrpd2bK-/view?usp=sharing).

SDSD-indoor:[[Google Drive]](https://drive.google.com/file/d/1U58zbbOvv_s_4ZWtDk_NssOcG1pqt3A7/view?usp=sharing).
SDSD-outdoor:[[Google Drive]](https://drive.google.com/file/d/1sJQSjxbfRb73k4JR7m6Yv3LzhD-BKy6D/view?usp=sharing).

SID:[[Google Drive]](https://drive.google.com/file/d/1dAxiSX_MeDmwVn1_m69qcoHwm6eaQ3Ni/view?usp=sharing).
SMID:[[Google Drive]](https://drive.google.com/file/d/1cB4xB1iJ3OSQk9STsYgxXeHlHi-ud5aj/view?usp=sharing).

Evaluation Metrics of GLTG-AFlow in Table 2 are obtained by pre-trained models via the following links

LOL-v2-real-zeromap:[[Google Drive]](https://drive.google.com/file/d/1wECxXPZ8Coc5KllMXSstGGXugtuE1I4l/view?usp=sharing).
LOL-v2-syn-zeromap:[[Google Drive]](https://drive.google.com/file/d/1DMFwJObvsGZKKip-G6DVd2lta0QtV0tF/view?usp=sharing).

SDSD-indoor-zeromap:[[Google Drive]](https://drive.google.com/file/d/1asfvlrD9dFARAf8lAoH1WeAKjjox4yDY/view?usp=sharing).
SDSD-outdoor-zeromap:[[Google Drive]](https://drive.google.com/file/d/1XysG63_hsZzPGPSOeqxfJ5GsPkAHzRZ8/view?usp=sharing).

SID-zeromap:[[Google Drive]](https://drive.google.com/file/d/1Voe0kcAeCDOcjDaT70e3dEgYNn3SGOZ-/view?usp=sharing).

Evaluation Metrics of GLTG-AFlow in Table 3 are obtained by pre-trained models via the following links

LOL-v2-real:[[Google Drive]](https://drive.google.com/file/d/1WIYhozFftZxjx1rx3LIN4Z_1QFE_JlER/view?usp=sharing).
LOL-v2-syn:[[Google Drive]](https://drive.google.com/file/d/1P9xkb_nJYL9W9IsY0vw8-seCIrpd2bK-/view?usp=sharing).

SDSD-indoor:[[Google Drive]](https://drive.google.com/file/d/1U58zbbOvv_s_4ZWtDk_NssOcG1pqt3A7/view?usp=sharing).
SDSD-outdoor:[[Google Drive]](https://drive.google.com/file/d/1sJQSjxbfRb73k4JR7m6Yv3LzhD-BKy6D/view?usp=sharing).

### Zero-map set
We construct a zero-map set with zero-maps from real-world outdoor night monitoring images, which are randomly combined with low-light images of public datasets, to form low-light images with massive zero-element pixels.

Before using ```test_with_zeromaps.py``` to restore low-light images with massive zero-element pixels, you need to download zero-map set.

Zero-map set:[[Google Drive]](https://drive.google.com/file/d/165Mx9sEYIyba9joK19B7o4MQlAq2WRcH/view?usp=sharing)

### Run the testing code 

You can test the model with low-light images and use 'Measure.py' to obtain the evaluation metrics. You need to specify the data path ```dataroot_test``` and model path ```model_path``` in the config file. Then run
```bash
python test.py
```

You can restore low-light images with massive zero-element pixels and use 'Measure.py' to obtain the evaluation metrics. You need to specify the model path ```model_path``` in the config file, put our zero-maps into ```.\Real_captured\Test\zero_map```, and put low-light image into ```.\Real_captured\Test\low```. Then run
```bash
python test_with_zeromaps.py
```

You can test the model with low-light images and use 'Measure.py' to obtain the evaluation metrics. You need to specify the data path ```dataroot_GT``` , ```dataroot_LR``` and model path ```model_path``` in the config file. Then run
```bash
python test_with_tricks.py
```