[TOC]

# 实验结果汇报

## Our setting (Knowledge Tracing)

- Data Preprocessing: The sequence length is fixed at 200, truncated to a maximum length and padded to a minimum length, with sequences shorter than 3 being discarded.

- Dataset Splitting: The dataset is divided by sequence, using 5-fold cross-validation to split into training and testing sets. Then, 20% of the training set in each fold is randomly selected as the validation set.

- Random Seed: All experiments fix the random seed to 0.

- Hyperparameter Selection: Hyperparameters are tuned on the 1st fold, the best hyperparameters are chosen based on the validation set, and then applied to the 5-fold cross-validation.

- Model Training Stopping Criteria: Early stopping. If the AUC metric on the validation set does not improve by more than 0.001 within 10 epochs, training is stopped. The maximum number of epochs is 200.

- Comparison Methods: After reproducing the results from the original paper under the experimental setup of the original paper, experiments are conducted under our setup. Hyperparameters are tuned within the same space as mentioned in the original paper.

- Reported Results: The average of the test set results under 5-fold cross-validation, with AUC as the evaluation metric.

- Run example/prepare_dataset/our_setting.py to obtain the partitioned dataset.

- All hyperparameters under this experimental setup can be found in the scripts located in example/scripts/our_setting.

- Based on our experience, knowledge tracing models are less sensitive to hyperparameters, so even with different experimental setups, the hyperparameters from this setup can be used to train the model without needing to retune them.

Obtain the partitioned data [获取划分好的数据](https://drive.google.com/drive/folders/1HYERnQYJz3diK1TZXhd_gJDL7eW1VnOk?usp=sharing)

### Overall metric (AUC)

- 常规指标，计算所有样本的性能，汇报AUC指标

|           |  Assist2009   |  Assist2012   |   Ednet-KT1   |  Statics2011  |    Xes3g5m    | Slemapy-Anatomy |
| :-------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-------------: |
|    DKT    |    0.7481     |    0.7337     |    0.6612     |    0.7113     |    0.7838     |     0.6838      |
|   DKVMN   |    0.7456     |    0.7217     |     0.668     |    0.7046     |    0.7748     |     0.6745      |
|   SAKT    |    0.7328     |     0.721     |    0.6642     |    0.6776     |    0.7791     |      0.676      |
|   LPKT    |    0.7682     |    0.7884     | <u>0.7394</u> | <u>0.8216</u> | <u>0.8257</u> |  <u>0.7365</u>  |
|   DIMKT   |    0.7647     |    0.7845     |    0.7154     |    0.8198     |  **0.8262**   |     0.7285      |
| SimpleKT  |    0.7853     |    0.7818     |    0.7342     |    0.8214     |    0.8198     |     0.7316      |
|   QIKT    |    0.7843     |    0.7753     |    0.7329     |  **0.8268**   |    0.8232     |     0.7234      |
| SparseKT  |    0.7782     |    0.7727     |    0.7302     |    0.8162     |    0.8153     |     0.7258      |
|   MIKT    |  **0.7886**   | <u>0.7902</u> |  **0.7436**   |    0.8213     |    0.8179     |   **0.7369**    |
|    AKT    | <u>0.7854</u> |  **0.7904**   |    0.7345     |    0.8193     |    0.8225     |     0.7288      |
|   qDKT    |    0.7684     |    0.7861     |    0.7354     |    0.8191     |    0.8252     |     0.7247      |
| AKT-CORE  |    0.7512     |    0.7619     |    0.7076     |    0.7811     |    0.8037     |     0.7133      |
| qDKT-CORE |    0.7365     |    0.7527     |    0.6544     |    0.7608     |     0.78      |     0.7008      |

|          | Assist2017 | Junyi2015 | Edi2020-task1 | Edi2020-task34 |
| :------: | :--------: | :-------: | :-----------: | :------------: |
|   qDKT   |   0.7919   |  0.7806   |    0.8141     |     0.7947     |
|   AKT    |   0.772    |  0.7791   |    0.8129     |     0.793      |
|   LPKT   |   0.812    |           |    0.8179     |     0.7968     |
|  DIMKT   |   0.8002   |  0.7836   |    0.8138     |     0.7936     |
| SimpleKT |   0.7746   |  0.7793   |    0.8135     |     0.7937     |
|   QIKT   |   0.7874   |  0.7812   |      OOM      |     0.7972     |

- 一些习题数量较少（基于习题建模，数据较稠密）的数据集上，基于习题的DKT和DKVMN

|                 | Assist2017 | Edi2020-task34 | Statics2011 | Xes3g5m |
| :-------------: | :--------: | :------------: | :---------: | :-----: |
|  DKT (concept)  |   0.7284   |     0.7598     |   0.7113    | 0.7838  |
|    DKT (que)    |   0.7953   |     0.7913     |   0.8145    | 0.8226  |
| DKVMN (concept) |   0.6941   |     0.748      |   0.7046    | 0.7748  |
|   DKVMN (que)   |   0.7411   |     0.7886     |   0.8032    | 0.8213  |

- LBKT在有行为数据的数据集上的性能

| Assist2009 | Assist2012 | Assist2017 | Junyi2015 |
| ---------- | ---------- | ---------- | --------- |
| 0.7767     | 0.7914     | 0.8335     | 0.7829    |

- KCQRL复现
  - Automated **K**nowledge **C**oncept Annotation and **Q**uestion **R**epresentation **L**earning for Knowledge Tracing （[paper](https://arxiv.org/abs/2410.01727), [code](https://github.com/oezyurty/KCQRL)）
  - direct que emb：直接使用习题文本的emb训练KT模型。其中Xes3g5m使用数据集提供的question emb，Moocradar-C_746997和Edi2020-task34调用ZhipuAI/embedding-3获取question emb
  - KCQRL emb：使用论文提出的训练方法所得到的question emb
  - 训练KT模型时，设置预训练的question emb为可学习，KT模型的参数都设置为和baseline一致
  - DIMKT：no concept表示concept emb和concept diff emb设置为0（和KCQRL论文一致）；use concept表示设置concept emb和concept diff emb（使用数据集提供的知识点）为可学习的embedding

|                                         | Xes3g5m | Moocradar-C_746997 | Edi2020-task34 |
| :-------------------------------------: | :-----: | :----------------: | :------------: |
|             DKT (baseline)              | 0.8226  |       0.8126       |     0.7877     |
|        DKT_QUE (direct que emb)         |  0.827  |       0.8205       |     0.7885     |
|           DKT_QUE (KCQRL emb)           |  0.828  |       0.8164       |     0.7888     |
|            DKVMN (baseline)             | 0.8213  |       0.8073       |     0.7868     |
|       DKVMN_QUE (direct que emb)        | 0.8232  |       0.8148       |     0.7912     |
|          DKVMN_QUE (KCQRL emb)          | 0.8232  |       0.8101       |     0.7907     |
|             AKT (baseline)              | 0.8225  |       0.8155       |     0.793      |
|        AKT_QUE (direct que emb)         | 0.8287  |       0.8202       |     0.7957     |
|           AKT_QUE (KCQRL emb)           | 0.8281  |       0.8187       |     0.7948     |
|            DIMKT (baseline)             | 0.8262  |       0.8186       |     0.7936     |
| DIMKT_QUE (no concept, direct que emb)  | 0.8259  |       0.8208       |     0.794      |
| DIMKT_QUE (use concept, direct que emb) | 0.8257  |       0.8205       |     0.7938     |
|    DIMKT_QUE (no concept, KCQRL emb)    | 0.8248  |       0.8211       |     0.7947     |

- KCQRL消融实验
  - KT模型参数和KCQRL一致，只改变输入的que emb
  - w/o step：对比学习训练习题的emb时，只使用知识点，不使用解题步骤
  - w/o cluster：对比学习训练习题的emb时，不对知识点进行聚类
  - (LLM)：使用LLM提取的知识点训练que emb
  - (KC)：使用数据集提供的知识点训练que emb
  - 训练que emb代码：[翻译和解题步骤获取代码](https://github.com/ZhijieXiong/DSPY-application/tree/main/dspy-project/KnowledgeTracing/KCQRL), [训练emb代码](https://github.com/ZhijieXiong/KCQRL-train-que-emb)
  

|           |           | KCQRL  | w/o step (LLM) | w/o step & cluster (LLM) | w/o step (KC) | w/o step & cluster (KC) |
| :-------: | :-------: | :----: | :------------: | :----------------------: | :-----------: | :---------------------: |
|  Xes3g5m  |  DKT_QUE  | 0.828  |     0.8271     |          0.8279          |    0.8273     |         0.8273          |
|  Xes3g5m  | DKVMN_QUE | 0.8232 |     0.8231     |          0.8229          |    0.8232     |         0.8227          |
|  Xes3g5m  |  AKT_QUE  | 0.8281 |     0.8282     |          0.8276          |    0.8272     |         0.8273          |
| Moocradar |  DKT_QUE  | 0.8164 |     0.8154     |          0.8169          |     0.817     |         0.8174          |
| Moocradar | DKVMN_QUE | 0.8101 |     0.8103     |          0.8088          |    0.8135     |         0.8128          |
| Moocradar |  AKT_QUE  | 0.8187 |     0.8173     |          0.8169          |     0.818     |         0.8185          |
|  Edi2020  |  DKT_QUE  | 0.7888 |     0.7886     |          0.7884          |    0.7891     |         0.7887          |
|  Edi2020  | DKVMN_QUE | 0.7907 |     0.7909     |          0.7912          |    0.7914     |         0.7917          |
|  Edi2020  |  AKT_QUE  | 0.7948 |     0.7946     |          0.795           |    0.7952     |         0.7958          |

### CORE metric (AUC)

- 论文`"Model-agnostic counterfactual reasoning for identifying and mitigating answer bias in knowledge tracing", Neural Networks 2024`提出来的一种无偏指标
- 汇报AUC指标

|          | Assist2009 | Assist2012 | Ednet-KT1 | Statics2011 | Xes3g5m | Slemapy-Anatomy |
| :------: | :--------: | :--------: | :-------: | :---------: | :-----: | :-------------: |
|   DKT    |   0.6931   |   0.6716   |  0.5857   |   0.6447    | 0.7031  |     0.6681      |
|  DKVMN   |   0.6859   |   0.6615   |  0.5817   |   0.6468    | 0.6979  |     0.6622      |
|   SAKT   |   0.6755   |   0.6582   |  0.5806   |   0.6283    | 0.7003  |     0.6651      |
|   LPKT   |   0.6559   |   0.6684   |  0.5712   |   0.6061    | 0.7097  |     0.6789      |
|  DIMKT   |   0.6821   |   0.664    |  0.5671   |   0.6131    | 0.7102  |     0.6679      |
| SimpleKT |   0.6903   |   0.6607   |  0.5722   |   0.6155    | 0.7002  |     0.6712      |
|   QIKT   |   0.6776   |   0.6469   |  0.5652   |   0.6262    | 0.7076  |     0.6634      |
| SparseKT |   0.6754   |   0.6438   |  0.5591   |   0.6025    | 0.6914  |     0.6667      |
|   MIKT   |   0.6874   |   0.6673   |  0.5809   |   0.6161    | 0.6895  |     0.6791      |
|   AKT    |   0.6955   |   0.6789   |  0.5769   |   0.6173    | 0.7127  |     0.6776      |
|   qDKT   |   0.6826   |   0.666    |  0.5708   |   0.6146    | 0.7078  |      0.665      |
| AKT-CORE |   0.6966   |   0.6965   |  0.5858   |   0.6319    | 0.7315  |     0.6902      |

## Our DG setting (Knowledge Tracing)

- 域泛化实验设置，具体如下
- Assist2009和Assist2012数据集有学生学校信息，所以可以基于学校做域泛化的实验，实验设置如下

  1、 合并人数少的学校为一个学校，同时不将极端学校（序列平均长度小于20）作为测试集数据。

  2、合并完成后，首先以学校为单位随机划分80%的学校为训练集，并且要求训练集的样本数量占总样本数量的70%～85%。

  3、划分训练集和测试集后，以学生为单位从训练集中划分20%作为验证集

  4、通过随机划分得到10种不同划分情况，用qDKT测量模型在验证集和测试集上的性能gap

  5、选择gap最大的结果做实验，为了降低随机性，汇报结果为5个随机种子的结果取平均

  6、模型停止训练的方法仍然是early stop，选择验证集性能最高的模型

  7、因为验证集是I.I.D的，所以各个模型的参数和our setting一样，并没有调参

  Slepemapy-Anatomy数据集有学生城市信息，所以可以基于城市做域泛化的实验，由于该数据集中其中一个城市的学生数据占比达到80%，所以直接使用该城市学生数据作为训练集，剩余数据作为测试集
- 汇报结果，使用AUC指标，括号外为验证集（I.I.D.）性能，括号内为测试集（O.O.D.）性能

|       | Assist2009      | Assist2012      | Slepemapy-Anatomy |
| ----- | --------------- | --------------- | ----------------- |
| qDKT  | 0.7482 (0.7327) | 0.7748 (0.7523) | 0.7258 (0.7096)   |
| AKT   | 0.7558 (0.7321) | 0.7766 (0.7506) | 0.7303 (0.7129)   |
| LPKT  | 0.7525 (0.7416) | 0.7787 (0.7577) | 0.7423 (0.7238)   |
| DIMKT | 0.7247 (0.7386) | 0.7724 (0.7449) | 0.7303 (0.7139)   |
| LBKT  | 0.7603 (0.7482) | 0.779 (0.7574)  |                   |

## pyKT setting (Knowledge Tracing)

- 要查看完整的实验记录，请点击[此处](https://docs.qq.com/sheet/DREdTc3lVQWJkdFJw?tab=BB08J2)。

- **调参是在 `example/prepare_datset/our_setting` 实验设置下进行的。我们在 `pykt_question_setting` 中直接使用了 `our_setting` 的参数，因此复现结果与论文中报告的结果略有不同。**，然后取5折的平均值（为了减少随机性，所有实验的随机种子都固定为0）。括号中的值是论文报告结果。表中的汇报指标为`AUC`。

- 论文中报告结果来自 `pyKT`、`SimpleKT`、`AT-DKT` 和 `QIKT`。请参阅[相应的论文](md_doc/MODELS.md)。

- Due to the data leakage issue in the original code of ATKT, we used the atktfix provided by pyKT.

- The table below shows the replication results on the multi-concept dataset. Please note:
  1. We did not, like pyKT, first expand the exercise sequence into a knowledge concept sequence (see Figure 2 in the pyKT paper), then train the model on the knowledge concept sequence, and finally test the model on the questions (see Section 3.3 of the pyKT paper). Our replication directly trains and tests the model on the question sequence, meaning that for multi-concept questions, we use mean pooling to handle multiple concept embeddings.
2. This difference is not only reflected in the model's training and testing but also in the data preprocessing. pyKT first expands the sequences, then splits them, fixing the length of each sequence to 200. However, we directly split the sequences and fix the sequence length to 200.

  |          |   Assist2009   |     AL2005     |     BD2006     |    Xes3g5m     |
  | :------: | :------------: | :------------: | :------------: | :------------: |
  |   DKT    | 0.756(0.7541)  | 0.8162(0.8149) | 0.7748(0.8015) | 0.7849(0.7852) |
  |   AKT    | 0.7911(0.7853) | 0.8169(0.8306) | 0.8162(0.8208) | 0.8231(0.8207) |
  | SimpleKT | 0.7906(0.7744) | 0.8426(0.8254) | 0.8144(0.816)  | 0.821(0.8163)  |
  |   QIKT   | 0.7907(0.7878) |      OOM       |      OOM       |      todo      |
  |   qDKT   |     0.7762     |     0.8363     |     0.8144     | 0.8261(0.8225) |

The table below shows the replication results on the single-concept dataset. Please note:
1. For datasets with fewer questions, our DKT and ATKT also provide results with questions as the entries.
2. For the statics2011 and edi2020-task34 datasets, our data preprocessing is different from pyKT.

|               |  Statics2011   | NIPS34 (Edi2020-task34) |
| :-----------: | :------------: | :---------------------: |
|      DKT      |     0.7142     |      0.762(0.7681)      |
|  DKT_use_que  | 0.8161(0.8222) |     0.7935(0.7995)      |
|     DKVMN     |     0.7066     |     0.7512(0.7673)      |
| DKVMN_use_que | 0.8078(0.8093) |         0.7901          |
|     SAINT     | 0.7273(0.7599) |     0.7846(0.7873)      |
|     ATKT      |     0.696      |     0.7603(0.7665)      |
| ATKT_use_que  | 0.8018(0.8055) |         0.7844          |
|      AKT      | 0.8244(0.8309) |     0.7943(0.8033)      |
|   SimpleKT    | 0.8258(0.8199) |     0.7955(0.8035)      |
|    AT-DKT     |      todo      |          todo           |
|     QIKT      |     0.8303     |     0.7993(0.8044)      |
|     qDKT      |     0.8236     |         0.7968          |

## Other Setting (Knowledge Tracing)

-To view the full experimental records, please click (https://docs.qq.com/sheet/DREtXSUtqTkZrTVVY?tab=BB08J2)

## NCD Setting (Cognitive Diagnosis)

- 论文：`"Neural Cognitive Diagnosis for Intelligent Education Systems"`	

| Assist2009 | AUC    | ACC    | RMSE   |
| ---------- | ------ | ------ | ------ |
| paper      | 0.749  | 0.719  | 0.439  |
| repro      | 0.7551 | 0.7236 | 0.4328 |

