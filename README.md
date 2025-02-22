# intrinsic-debiasing-performance-on-NMT
This repo contains code and data for reproducing the experiments in [The Impact of Intrinsic Debiasing on Downstream Tasks: A Case Study on Machine Translation](https://arxiv.org/pdf/2406.00787) [Bar Iluz](https://arxiv.org/search/cs?searchtype=author&query=Iluz,+B), Yanai Elazar,  Asaf Yehudai, [Gabriel Stanovsky](https://gabrielstanovsky.github.io/), (EMNLP 2024)

## Citing

```
@article{iluz2024applying,
  title={Applying Intrinsic Debiasing on Downstream Tasks: Challenges and Considerations for Machine Translation},
  author={Iluz, Bar and Elazar, Yanai and Yehudai, Asaf and Stanovsky, Gabriel},
  journal={arXiv preprint arXiv:2406.00787},
  year={2024}
}
```
## Requirements:
Requirements file:
```
pip install -r requirements.txt
```

The code uses several translation models and debiasing methods.
For the chosen configuration to work you need to clone the following repositories forks:
Translation models:
  Nematus:
  [nematus bariluz github](https://github.com/bariluz93/nematus/tree/switch_to_easy_nmt)
  ```
  git clone https://github.com/bariluz93/nematus.git
  ```
  Opus-MT and Mbart50:
  [easynmt bariluz github](https://github.com/bariluz93/EasyNMT)
  ```
  https://github.com/bariluz93/EasyNMT.git
  ```
Debiasing methods:
  Hard-Debias:
  [Hard-Debias bariluz github](https://github.com/bariluz93/debiaswe)
  ```
  git clone https://github.com/bariluz93/debiaswe.git
  ```
  INLP:
  [NullSpace Projection bariluz github](https://github.com/bariluz93/nullspace_projection)
  ```
  https://github.com/bariluz93/nullspace_projection.git
  ```
  LEACE:
  Install concept erasure
  ```
  pip install concept-erasure
  ```

  
  


The code will be updated soon
