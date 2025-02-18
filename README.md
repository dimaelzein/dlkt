[Document] | [Datasets]  | [Experiement]

[Document]: md_doc/DOC.md
[Datasets]: md_doc/DATASETS.md
[Experiement]: md_doc/Experiement.md
# Introduction

A library of algorithms for reproducing knowledge tracing, cognitive diagnosis, and exercise recommendation models.

# Quick-Start

## Prepare

1. Initialize project

   - Create file `settings.json` in the root directory.

   - Modify the environment configuration file `settings.json`

     ```python
     {
       "LIB_PATH": ".../dlkt-main",  # Change to the project root path
       "FILE_MANAGER_ROOT": "any_dir"  # Any path used to store data and models
     }
     ```

   - Run `set_up.py`

     ```shell
     python set_up.py
     ```

2. Place the original files of the dataset in the corresponding directory (Please refer to [Document (Section 1.3)](md_doc/DOC.md) for details)

3. Data Preprocessing: Run ` example/preprocess.py`, for example

   ```shell
   python preprocess.py --dataset_name assist2009
   ```

## Knowledge Tracing

1. Divide the dataset according to the specified experimental settings: Run `example4knowledge_racing/prepare_dataset/akt_setting.py`. For example, 

   ```shell
   python akt_setting.py
   ```

   - For details on dataset partitioning, please refer to [Document (Section 1.6)](md_doc/DOC.md)

2. Train model: Run the file under `example/train`. For example, train a DKT model

   ```shell
   python dkt.py
   ```

   - Regarding the meaning of parameters, please refer to [Document (Section 2)](Doc.md)

## Cognitive Diagnosis

1. Divide the dataset according to the specified experimental settings: Run `example4cognitive_diagnosis/prepare_dataset/akt_setting.py`. For example, 

   ```shell
   python ncd_setting.py
   ```

2. Train model: Run the file under `example4cognitive_diagnosis/train`. For example, train a NCD model

   ```shell
   python ncd.py
   ```

## Exercise Recommendation

1. Divide the dataset according to the specified experimental settings: Run `example4exercise_recommendation/prepare_dataset/kg4ex_setting.py`. For example, 

   ```
   python kg4ex_setting.py
   ```

2. Train or evaluate different model or method

   1. KG4EX
      - step 1, train a `DKT` model to get mlkc
      - step 2, train a `DKT_KG4EX` model to get pkc
      - step 3, run `example4exercise_recommendation/kg4ex/get_mlkc_pkc.py`
      - step 4, run `example4exercise_recommendation/kg4ex/get_efr.py`
      - step 5, run `example4exercise_recommendation/kg4ex/get_triples.py`
      - step 6, run `example4exercise_recommendation/train/kg4ex.py`

   2. EB-CF (Exercise-based collaborative filtering)
      - step1, change `example4exercise_recommendation/eb_cf/load_data` to get users' history data
      - step2, run `example4exercise_recommendation/eb_cf/get_que_sim_mat.py` to get questions' similarity matrix
      - step3, run `example4exercise_recommendation/eb_cf/evaluate.py`


# Referrence

- code
  - https://github.com/pykt-team/pykt-toolkit
  - https://github.com/bigdata-ustc/EduKTM
  - https://github.com/bigdata-ustc/EduData
  - https://github.com/bigdata-ustc/EduCDM
- paper
  - [*Chris Piech, Jonathan Bassen, Jonathan Huang, Surya Ganguli, Mehran Sahami, Leonidas J. Guibas, Jascha Sohl-Dickstein*. **Deep Knowledge Tracing**. Advances in Neural Information Processing Systems 28 (NIPS 2015)](https://proceedings.neurips.cc/paper_files/paper/2015/hash/bac9162b47c56fc8a4d2a519803d51b3-Abstract.html)
  - [*Jiani Zhang, Xingjian Shi, Irwin King, and Dit-Yan Yeung*. **Dynamic Key-Value Memory Networks for Knowledge Tracing**. In Proceedings of the 26th International Conference on World Wide Web (WWW '17)](https://dl.acm.org/doi/abs/10.1145/3038912.3052580)
  - [*Shalini Pandey, George Karypis*. **A Self-Attentive model for Knowledge Tracing**. EDM 2019 - Proceedings of the 12th International Conference on Educational Data Mining](https://experts.umn.edu/en/publications/a-self-attentive-model-for-knowledge-tracing)
  - [*Hiromi Nakagawa, Yusuke Iwasawa, and Yutaka Matsuo*. **Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network**. In IEEE/WIC/ACM International Conference on Web Intelligence (WI '19).](https://dl.acm.org/doi/abs/10.1145/3350546.3352513)
  - [*Youngduck Choi, Youngnam Lee, Junghyun Cho, Jineon Baek, Byungsoo Kim, Yeongmin Cha, Dongmin Shin, Chan Bae, and Jaewe Heo*. 2020. **Towards an Appropriate Query, Key, and Value Computation for Knowledge Tracing**. In Proceedings of the Seventh ACM Conference on Learning @ Scale (L@S '20).](https://dl.acm.org/doi/abs/10.1145/3386527.3405945)
  - [*Sonkar, Shashank, et al*. **qdkt: Question-centric deep knowledge tracing**. arXiv preprint arXiv:2005.12442 (2020)](https://arxiv.org/pdf/2005.12442.pdf)
  - [*Yang Yang, Jian Shen, Yanru Qu, Yunfei Liu, Kerong Wang, Yaoming Zhu, Weinan Zhang, Yong Yu*. **GIKT: A Graph-Based Interaction Model for Knowledge Tracing**. ECML PKDD 2020, LNAI 12457, pp. 299–315, 2021.](https://link.springer.com/chapter/10.1007/978-3-030-67658-2_18)
  - [*Aritra Ghosh, Neil Heffernan, and Andrew S. Lan*. **Context-Aware Attentive Knowledge Tracing**. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '20)](https://doi.org/10.1145/3394486.3403282)
  - [*Seewoo Lee , Youngduck Choi , Juneyoung Park, Byungsoo Kim1 and Jinwoo Shin*. **Consistency and Monotonicity Regularization for Neural Knowledge Tracing**. arXiv:2105.00607](https://arxiv.org/abs/2105.00607)
  - [*Shuanghong Shen, Qi Liu, Enhong Chen, Zhenya Huang, Wei Huang, Yu Yin, Yu Su, and Shijin Wang*. **Learning Process-consistent Knowledge Tracing**. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining (KDD '21)](https://dl.acm.org/doi/abs/10.1145/3447548.3467237)
  - [*Ting Long, Yunfei Liu, Jian Shen, Weinan Zhang, and Yong Yu*. **Tracing Knowledge State with Individual Cognition and Acquisition Estimation**. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '21).](https://dl.acm.org/doi/abs/10.1145/3404835.3462827)
  - [*Xiaopeng Guo, Zhijie Huang, Jie Gao, Mingyu Shang, Maojing Shu, and Jun Sun.* **Enhancing Knowledge Tracing via Adversarial Training**. In Proceedings of the 29th ACM International Conference on Multimedia (MM '21)](https://dl.acm.org/doi/abs/10.1145/3474085.3475554)
  - [*Shuanghong Shen, Zhenya Huang, Qi Liu, Yu Su, Shijin Wang, and Enhong Chen*. **Assessing Student's Dynamic Knowledge State by Exploring the Question Difficulty Effect**. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '22)](https://dl.acm.org/doi/abs/10.1145/3477495.3531939)
  - [*Wonsung Lee, Jaeyoon Chun, Youngmin Lee, Kyoungsoo Park, and Sungrae Park*. **Contrastive Learning for Knowledge Tracing**. In Proceedings of the ACM Web Conference 2022 (WWW '22)](https://dl.acm.org/doi/abs/10.1145/3485447.3512105)
  - [*Minn, S., Vie, J.-J., Takeuchi, K., Kashima, H. and Zhu, F*. **Interpretable Knowledge Tracing: Simple and Efficient Student Modeling with Causal Relations**. *Proceedings of the AAAI Conference on Artificial Intelligence* (AAAI-22).](https://ojs.aaai.org/index.php/AAAI/article/view/21560)
  - [*Zitao Liu, Qiongqiong Liu, Jiahao Chen, Shuyan Huang,Weiqi Luo*. **simpleKT: A Simple But Tough-to-Beat Baseline for Knowledge Tracing**. The Eleventh International Conference on Learning Representations. 2022](https://openreview.net/forum?id=9HiGqC9C-KA)
  - [*Yu Yin, Le Dai, Zhenya Huang, Shuanghong Shen, Fei Wang, Qi Liu, Enhong Chen, and Xin Li*. **Tracing Knowledge Instead of Patterns: Stable Knowledge Tracing with Diagnostic Transformer**. In Proceedings of the ACM Web Conference 2023 (WWW '23)](https://dl.acm.org/doi/abs/10.1145/3543507.3583255)
  - [*Zitao Liu, Qiongqiong Liu, Jiahao Chen, Shuyan Huang, Boyu Gao, Weiqi Luo, and Jian Weng*. **Enhancing Deep Knowledge Tracing with Auxiliary Tasks**. In Proceedings of the ACM Web Conference 2023 (WWW '23)](https://dl.acm.org/doi/abs/10.1145/3543507.3583866)
  - [*Chen, J., Liu, Z., Huang, S., Liu, Q. and Luo, W*. **Improving Interpretability of Deep Sequential Knowledge Tracing Models with Question-centric Cognitive Representations**. The Thirty-Seventh AAAI Conference on Artificial Intelligence (AAAI-23)](https://ojs.aaai.org/index.php/AAAI/article/view/26661)
  - [*Bihan Xu, Zhenya Huang, Jiayu Liu, Shuanghong Shen, Qi Liu, Enhong Chen, Jinze Wu, and Shijin Wang*. **Learning Behavior-oriented Knowledge Tracing**. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '23).](https://dl.acm.org/doi/abs/10.1145/3580305.3599407)
  - [*Chaoran Cuia , Hebo Ma, et al*. **Do We Fully Understand Students’ Knowledge States? Identifying and Mitigating Answer Bias in Knowledge Tracing**. arXiv:2308.07779](https://arxiv.org/pdf/2308.07779.pdf)
  - [*Jianwen Sun, Fenghua Yu, Qian Wan, Qing Li, Sannyuya Liu, and Xiaoxuan Shen*. **Interpretable Knowledge Tracing with Multiscale State Representation**. In Proceedings of the ACM on Web Conference 2024 (WWW '24).](https://dl.acm.org/doi/10.1145/3589334.3645373)
  - [*Wang, F., Liu, Q., Chen, E., Huang, Z., Chen, Y., Yin, Y., Huang, Z. and Wang, S*. **Neural Cognitive Diagnosis for Intelligent Education Systems**. *Proceedings of the AAAI Conference on Artificial Intelligence* (AAAI-20)](https://ojs.aaai.org/index.php/AAAI/article/view/6080)
  - [*Quanlong Guan, Fang Xiao, Xinghe Cheng, Liangda Fang, Ziliang Chen, Guanliang Chen, and Weiqi Luo*. **KG4Ex: An Explainable Knowledge Graph-Based Approach for Exercise Recommendation**. *In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management (CIKM '23)](https://dl.acm.org/doi/10.1145/3583780.3614943)


# Contributing

Please let us know if you encounter a bug or have any suggestions by [filing an issue](https://github.com/ZhijieXiong/dlkt/issuesWe) 

We welcome all contributions from bug fixes to new features and extensions.

We expect all contributions discussed in the issue tracker and going through PRs.

# Contributors

- https://github.com/ZhijieXiong
- https://github.com/kingofpop625
- https://github.com/shshen-closer
