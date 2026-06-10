# Paper

내가 읽은 논문 목록입니다. 각 항목은 `제목 | 연도 | 링크 | 요약`으로 정리하며, 카테고리 안에서는 **연도 오름차순**으로 정렬합니다.

- **Year**: 출판/공개 연도 (arXiv 게재 연도 기준 교차검증, 일부는 근사치)
- **Link**: arXiv 게재 논문은 arXiv 직접 링크, 그 외(비arXiv·미상)는 Google Scholar 검색 링크
- **Summary**: 한 줄 핵심 요약

## Categories
- [Machine Learning / Deep Learning](#machine-learning--deep-learning)
- [Continual Learning](#continual-learning)
- [Speech Recognition — Self-supervised & Representation Learning](#speech-recognition--self-supervised--representation-learning)
- [Speech Recognition — Architectures & Encoders](#speech-recognition--architectures--encoders)
- [Speech Recognition — CTC, Transducer & Objectives](#speech-recognition--ctc-transducer--objectives)
- [Speech Recognition — Streaming, Online & VAD](#speech-recognition--streaming-online--vad)
- [Speech Recognition — Decoding & LM Integration](#speech-recognition--decoding--lm-integration)
- [Speech Recognition — Accented, Dialect & Robust](#speech-recognition--accented-dialect--robust)
- [Speech Recognition — Efficiency & On-device](#speech-recognition--efficiency--on-device)
- [Speech Recognition — Large-scale, Weak & Semi-supervised](#speech-recognition--large-scale-weak--semi-supervised)
- [Speech Recognition — Continual Learning](#speech-recognition--continual-learning)
- [Speech Recognition — Multilingual](#speech-recognition--multilingual)
- [Speech Recognition — Code-switching](#speech-recognition--code-switching)
- [Speech Recognition — General, Surveys & Features](#speech-recognition--general-surveys--features)
- [Pronunciation Assessment & L2 Learning (CAPT)](#pronunciation-assessment--l2-learning-capt)
- [Speech LLM / Audio LLM](#speech-llm--audio-llm)
- [Speaker Recognition & Diarization](#speaker-recognition--diarization)
- [Speech Enhancement](#speech-enhancement)
- [Speech Augmentation](#speech-augmentation)
- [Speech Toolkit](#speech-toolkit)
- [Speech Dataset](#speech-dataset)
- [Spoken Dialogue & Full-duplex (S2S)](#spoken-dialogue--full-duplex-s2s)
- [TTS / Voice Conversion](#tts--voice-conversion)
- [NLP / LLM](#nlp--llm)
- [Computer Vision](#computer-vision)
- [Multimodal](#multimodal)
- [Reinforcement Learning](#reinforcement-learning)
- [Linguistics](#linguistics)

## Machine Learning / Deep Learning
| Title | Year | Link | Summary |
|-------|------|------|---------|
| Long-short Term Memory | 1997 | [link](https://scholar.google.com/scholar?q=Long-short+Term+Memory) | 게이트로 장기 의존성 학습하는 LSTM |
| Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks | 2013 | [link](https://scholar.google.com/scholar?q=Pseudo-label:+The+simple+and+efficient+semi-supervised+learning+method+for+deep+neural+networks) | 모델 예측을 가짜 라벨로 쓰는 준지도학습 |
| Generative Adversarial Nets | 2014 | [arXiv](https://arxiv.org/abs/1406.2661) | 생성자-판별자 적대 학습(GAN) 제안 |
| Learning with Pseudo-Ensembles | 2014 | [arXiv](https://arxiv.org/abs/1412.4864) | 노이즈 섭동에 일관된 pseudo-ensemble 학습 |
| Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift | 2015 | [arXiv](https://arxiv.org/abs/1502.03167) | 미니배치 정규화로 학습 가속·안정화 |
| Distilling the Knowledge in a Neural Network | 2015 | [arXiv](https://arxiv.org/abs/1503.02531) | soft target으로 지식 증류(KD) 제안 |
| Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding | 2015 | [arXiv](https://arxiv.org/abs/1510.00149) | 가지치기+양자화+허프만으로 모델 압축 |
| Deep Networks with Stochastic Depth | 2016 | [arXiv](https://arxiv.org/abs/1603.09382) | 학습 시 레이어 확률적 드롭으로 정규화 |
| Domain-Adversarial Training of Neural Networks | 2015 | [arXiv](https://arxiv.org/abs/1505.07818) | gradient reversal로 도메인 불변 표현 학습(DANN) |
| Gaussian Error Linear Units (GELUs) | 2016 | [arXiv](https://arxiv.org/abs/1606.08415) | 확률적 게이팅 기반 활성함수 GELU |
| Layer Normalization | 2016 | [arXiv](https://arxiv.org/abs/1607.06450) | 피처 차원 정규화, RNN에 적합 |
| SGDR: Stochastic Gradient Descent with Warm Restarts | 2016 | [arXiv](https://arxiv.org/abs/1608.03983) | 코사인 스케줄+warm restart 학습률 |
| An overview of Multi-Task Learning in Deep Neural Networks | 2017 | [arXiv](https://arxiv.org/abs/1706.05098) | 멀티태스크 학습 개념·기법 개요 |
| Mixed Precision Training | 2017 | [arXiv](https://arxiv.org/abs/1710.03740) | FP16+loss scaling으로 학습 메모리·속도 개선 |
| Monotonic Chunkwise Attention | 2017 | [arXiv](https://arxiv.org/abs/1712.05382) | 단조 정렬+청크 attention으로 온라인 디코딩(MoChA) |
| Online and Linear-Time Attention by Enforcing Monotonic Alignments | 2017 | [arXiv](https://arxiv.org/abs/1704.00784) | 단조 정렬 강제로 선형시간 온라인 attention |
| Searching for Activation Functions | 2017 | [arXiv](https://arxiv.org/abs/1710.05941) | 탐색으로 찾은 활성함수 Swish |
| A Survey on Deep Transfer Learning | 2018 | [arXiv](https://arxiv.org/abs/1808.01974) | 딥 전이학습 분류와 기법 서베이 |
| Active Learning for Convolutional Neural Networks: A Core-Set Approach | 2017 | [arXiv](https://arxiv.org/abs/1708.00489) | core-set 선택으로 CNN 라벨링 효율화 |
| Group Normalization | 2018 | [arXiv](https://arxiv.org/abs/1803.08494) | 채널 그룹 단위 정규화, 배치 크기 무관 |
| Learning Loss for Active Learning | 2019 | [arXiv](https://arxiv.org/abs/1905.03677) | 손실값 예측 모듈로 unlabeled 샘플 선택 |
| Unsupervised Data Augmentation for Consistency Training | 2019 | [arXiv](https://arxiv.org/abs/1904.12848) | 강한 증강에 일관성 부여하는 준지도학습(UDA) |
| ZeRO: Memory optimizations Toward Training Trillion Parameter Models | 2019 | [arXiv](https://arxiv.org/abs/1910.02054) | 옵티마이저/그래디언트 분할로 초대형 모델 학습 |
| A Survey of Deep Active Learning | 2020 | [arXiv](https://arxiv.org/abs/2009.00236) | 딥 액티브 러닝 전반 서베이 |
| A Comparative Survey of Deep Active Learning | 2022 | [arXiv](https://arxiv.org/abs/2203.13450) | 딥 액티브 러닝 방법론 비교 서베이 |

## Continual Learning
| Title | Year | Link | Summary |
|-------|------|------|---------|
| Learning without Forgetting | 2016 | [arXiv](https://arxiv.org/abs/1606.09282) | 옛 태스크 출력 보존하며 새 태스크 학습(LwF) |
| Progressive Neural Networks | 2016 | [arXiv](https://arxiv.org/abs/1606.04671) | 태스크별 컬럼 추가로 망각 방지 |
| Gradient Episodic Memory for Continual Learning | 2017 | [arXiv](https://arxiv.org/abs/1706.08840) | 과거 손실 증가 막는 gradient 제약(GEM) |
| Overcoming catastrophic forgetting in neural networks | 2017 | [arXiv](https://arxiv.org/abs/1612.00796) | 파라미터 중요도 기반 정규화(EWC) |

## Speech Recognition — Self-supervised & Representation Learning
| Title | Year | Link | Summary |
|-------|------|------|---------|
| Representation Learning with Contrastive Predictive Coding | 2018 | [arXiv](https://arxiv.org/abs/1807.03748) | InfoNCE로 미래 latent를 대조 예측하는 범용 SSL(CPC) |
| vq-wav2vec: Self-Supervised Learning of Discrete Speech Representations | 2019 | [arXiv](https://arxiv.org/abs/1910.05453) | 음성을 이산 토큰으로 양자화해 BERT식 사전학습 가능케 함 |
| wav2vec: Unsupervised Pre-training for Speech Recognition | 2019 | [arXiv](https://arxiv.org/abs/1904.05862) | CNN 기반 contrastive 사전학습으로 저자원 ASR 개선 |
| wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations | 2020 | [arXiv](https://arxiv.org/abs/2006.11477) | Transformer + quantization + contrastive로 라벨 적은 ASR SOTA |
| HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units | 2021 | [arXiv](https://arxiv.org/abs/2106.07447) | k-means hidden unit을 마스킹 예측해 음성 표현 학습 |
| W2v-BERT: Combining Contrastive Learning and Masked Language Modeling for Self-Supervised Speech Pre-Training | 2021 | [arXiv](https://arxiv.org/abs/2108.06209) | contrastive + MLM을 end-to-end로 동시 최적화 |
| data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language | 2022 | [arXiv](https://arxiv.org/abs/2202.03555) | 음성·비전·언어를 단일 프레임워크로, 마스킹 입력으로 teacher의 contextualized latent 예측 |
| Self-Supervised Learning with Random-Projection Quantizer for Speech Recognition | 2022 | [arXiv](https://arxiv.org/abs/2202.01855) | 학습 없는 random-projection quantizer로 마스킹 예측(BEST-RQ) |
| NEST: Self-supervised Fast Conformer as All-purpose Seasoning to Speech Processing Tasks | 2024 | [arXiv](https://arxiv.org/abs/2408.13106) | FastConformer + random-projection으로 ASR·diarization·SLU 범용 SSL |

## Speech Recognition — Architectures & Encoders
| Title | Year | Link | Summary |
|-------|------|------|---------|
| A time delay neural network architecture for efficient modeling of long temporal contexts | 2015 | [link](https://scholar.google.com/scholar?q=A+time+delay+neural+network+architecture+for+efficient+modeling+of+long+temporal+contexts) | 서브샘플링 TDNN으로 긴 문맥 효율 모델링 |
| Deep Speech2: End-to-end Speech Recognition in English and Mandarin | 2015 | [arXiv](https://arxiv.org/abs/1512.02595) | RNN+CTC 대규모 E2E ASR(영·중) |
| Listen, Attend and Spell | 2015 | [arXiv](https://arxiv.org/abs/1508.01211) | attention 기반 seq2seq ASR(LAS) |
| Wav2Letter: an End-to-End ConvNet-based Speech Recognition System | 2016 | [arXiv](https://arxiv.org/abs/1609.03193) | conv+ASG 손실 E2E ASR |
| Light Gated Recurrent Units for Speech Recognition | 2018 | [arXiv](https://arxiv.org/abs/1803.10225) | reset gate 제거한 경량 GRU(Li-GRU) |
| Speech-transformer: a no-recurrence sequence-to-sequence model for speech recognition | 2018 | [link](https://scholar.google.com/scholar?q=Speech-transformer:+a+no-recurrence+sequence-to-sequence+model+for+speech+recognition) | 순환 없는 Transformer seq2seq ASR |
| Jasper: An End-to-End Convolutional Neural Acoustic Model | 2019 | [arXiv](https://arxiv.org/abs/1904.03288) | 1D conv 블록 깊게 쌓은 E2E 음향모델 |
| Output-Gate Projected Gated Recurrent Unit for Speech Recognition | 2019 | [link](https://scholar.google.com/scholar?q=Output-Gate+Projected+Gated+Recurrent+Unit+for+Speech+Recognition) | 출력 게이트·프로젝션 추가한 GRU(OPGRU) |
| Quartznet: Deep Automatic Speech Recognition with 1D Time-Channel Separable Convolutions | 2019 | [arXiv](https://arxiv.org/abs/1910.10261) | 분리형 1D conv로 경량 고성능 ASR |
| Conformer: Convolution-augmented Transformer for Speech Recognition | 2020 | [arXiv](https://arxiv.org/abs/2005.08100) | conv+attention 결합으로 ASR 인코더 SOTA |
| ContextNet: Improving Convolutional Neural Networks for Automatic Speech Recognition with Global Context | 2020 | [arXiv](https://arxiv.org/abs/2005.03191) | SE 모듈로 전역 문맥 더한 CNN 인코더 |
| Efficient conformer: Progressive downsampling and grouped attention for automatic speech recognition | 2021 | [arXiv](https://arxiv.org/abs/2109.01163) | 점진 다운샘플+그룹 attention으로 경량 Conformer |
| Branchformer: Parallel MLP-Attention Architectures to Capture Local and Global Context for Speech Recognition and Understanding | 2022 | [arXiv](https://arxiv.org/abs/2207.02971) | attention과 cgMLP 병렬 분기로 지역·전역 포착 |
| E-Branchformer: Branchformer with Enhanced Merging for Speech Recognition | 2022 | [arXiv](https://arxiv.org/abs/2210.00077) | 분기 병합 개선한 Branchformer |
| Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition | 2022 | [arXiv](https://arxiv.org/abs/2206.08317) | 토큰 수 예측+병렬 디코딩 NAR ASR |
| Squeezeformer: An Efficient Transformer for Automatic Speech Recognition | 2022 | [arXiv](https://arxiv.org/abs/2206.00888) | U-Net식 다운/업샘플로 Conformer 효율화 |
| A Comparative Study on E-Branchformer vs Conformer in Speech Recognition, Translation, and Understanding Tasks | 2023 | [link](https://scholar.google.com/scholar?q=A+Comparative+Study+on+E-Branchformer+vs+Conformer+in+Speech+Recognition,+Translation,+and+Understanding+Tasks) | E-Branchformer와 Conformer 인코더 비교 |
| Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition | 2023 | [arXiv](https://arxiv.org/abs/2305.05084) | 8x 서브샘플+선형 attention 고속 Conformer |
| Multi-Head State Space Model for Speech Recognition | 2023 | [arXiv](https://arxiv.org/abs/2305.12498) | 멀티헤드 SSM 인코더로 ASR |
| Zipformer: A faster and better encoder for automatic speech recognition | 2023 | [arXiv](https://arxiv.org/abs/2310.11230) | 다해상도 U-Net 구조+ScaledAdam 인코더 |
| Samba-ASR: State-Of-The-Art Speech Recognition Leveraging Structured State-Space Models | 2025 | [arXiv](https://arxiv.org/abs/2501.02832) | Mamba 기반 SSM 인코더·디코더 ASR |

## Speech Recognition — CTC, Transducer & Objectives
| Title | Year | Link | Summary |
|-------|------|------|---------|
| Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks | 2006 | [link](https://scholar.google.com/scholar?q=Connectionist+Temporal+Classification:+Labelling+Unsegmented+Sequence+Data+with+Recurrent+Neural+Networks) | 정렬 없는 시퀀스 라벨링 손실(CTC) |
| Sequence Transduction with Recurrent Neural Networks | 2012 | [arXiv](https://arxiv.org/abs/1211.3711) | RNN-Transducer 원논문 |
| Purely sequence-trained neural networks for ASR based on lattice-free MMI | 2016 | [link](https://scholar.google.com/scholar?q=Purely+sequence-trained+neural+networks+for+ASR+based+on+lattice-free+MMI) | lattice 없이 시퀀스 판별 학습(LF-MMI) |
| Joint CTC-attention based end-to-end speech recognition using multi-task learning | 2016 | [arXiv](https://arxiv.org/abs/1609.06773) | CTC+attention 하이브리드 멀티태스크 |
| End-to-end Speech Recognition Using Lattice-free MMI | 2018 | [link](https://scholar.google.com/scholar?q=End-to-end+Speech+Recognition+Using+Lattice-free+MMI) | flat-start LF-MMI E2E 학습 |
| Mask CTC: Non-Autoregressive End-to-End ASR with CTC and Mask Predict | 2020 | [arXiv](https://arxiv.org/abs/2005.08700) | CTC 초안+마스크 예측 정제 NAR ASR |
| Rnn-Transducer with Stateless Prediction Network | 2020 | [link](https://scholar.google.com/scholar?q=Rnn-Transducer+with+Stateless+Prediction+Network) | 예측망을 무상태(임베딩)로 단순화 |
| Transformer transducer: A streamable speech recognition model with transformer encoders and rnn-t loss | 2020 | [arXiv](https://arxiv.org/abs/2002.02562) | Transformer 인코더+RNN-T로 스트리밍 |
| Intermediate Loss Regularization for CTC-based Speech Recognition | 2021 | [arXiv](https://arxiv.org/abs/2102.03216) | 중간 레이어에 보조 CTC 손실(InterCTC) |
| Accelerating RNN-T Training and Inference Using CTC guidance | 2022 | [arXiv](https://arxiv.org/abs/2210.16481) | CTC 정렬로 RNN-T 학습·추론 가속 |
| Memory-Efficient Training of RNN-Transducer with Sampled Softmax | 2022 | [arXiv](https://arxiv.org/abs/2203.16868) | sampled softmax로 RNN-T 메모리 절감 |
| Pruned RNN-T for fast, memory-efficient ASR training | 2022 | [arXiv](https://arxiv.org/abs/2206.13236) | lattice 가지치기로 RNN-T 손실 고속화 |
| Efficient Sequence Transduction by Jointly Predicting Tokens and Durations | 2023 | [arXiv](https://arxiv.org/abs/2304.06795) | 토큰+지속시간 동시 예측 transducer(TDT) |
| CR-CTC: Consistency regularization on CTC for improved speech recognition | 2024 | [arXiv](https://arxiv.org/abs/2410.05101) | 두 증강 뷰 간 CTC 일관성 정규화 |

## Speech Recognition — Streaming, Online & VAD
| Title | Year | Link | Summary |
|-------|------|------|---------|
| Towards Online End-to-end Transformer Automatic Speech Recognition | 2019 | [arXiv](https://arxiv.org/abs/1910.11871) | 온라인 Transformer ASR 초기 연구 |
| Improved end-of-query detection for streaming speech recognition | 2020 | [link](https://scholar.google.com/scholar?q=Improved+end-of-query+detection+for+streaming+speech+recognition) | 발화 종료 검출 개선으로 응답 지연 단축 |
| MarbleNet: Deep 1D Time-Channel Separable Convolutional Neural Network for Voice Activity Detection | 2020 | [arXiv](https://arxiv.org/abs/2010.13886) | 경량 분리형 conv VAD |
| Streaming Automatic Speech Recognition with the Transformer Models | 2020 | [arXiv](https://arxiv.org/abs/2001.02674) | 청크/룩어헤드로 Transformer 스트리밍화 |
| Streaming Transformer Asr With Blockwise Synchronous Beam Search | 2020 | [arXiv](https://arxiv.org/abs/2006.14941) | 블록 동기 빔서치 스트리밍 Transformer |
| Towards Fast and Accurate Streaming End-To-End ASR | 2020 | [arXiv](https://arxiv.org/abs/2004.11544) | RNN-T 기반 저지연·고정확 스트리밍 |
| E2E Segmentation in a Two-Pass Cascaded Encoder ASR Model | 2021 | [link](https://scholar.google.com/scholar?q=E2E+Segmentation+in+a+Two-Pass+Cascaded+Encoder+ASR+Model) | 2-pass 캐스케이드 인코더 내 세그멘테이션 |
| Emformer: Efficient Memory Transformer Based Acoustic Model for Low Latency Streaming Speech Recognition | 2020 | [arXiv](https://arxiv.org/abs/2010.10759) | 메모리 캐리오버 블록 attention 저지연 ASR |
| FastEmit: Low-latency Streaming ASR with Sequence-level Emission Regularization | 2020 | [arXiv](https://arxiv.org/abs/2010.11148) | 빠른 토큰 방출 유도하는 정규화 |
| VAD-free Streaming Hybrid CTC/Attention ASR for Unsegmented Recording | 2021 | [arXiv](https://arxiv.org/abs/2107.07509) | VAD 없이 비분할 녹음 스트리밍 디코딩 |
| E2E Segmenter: Joint Segmenting and Decoding for Long-Form ASR | 2022 | [arXiv](https://arxiv.org/abs/2204.10749) | 긴 발화 분할·디코딩 공동 학습 |
| End-to-End Automatic Speech Recognition Integrated with CTC-Based Voice Activity Detection | 2020 | [arXiv](https://arxiv.org/abs/2002.00551) | CTC blank로 VAD 통합한 E2E ASR |
| Reducing the Offline-Streaming Gap for Unified ASR Transducer with Consistency Regularization | 2026 | [arXiv](https://arxiv.org/abs/2604.19079) | 일관성 정규화로 오프라인-스트리밍 격차 축소 |
| Run-and-Back Stitch Search: Novel Block Synchronous Decoding For Streaming Encoder-Decoder ASR | 2022 | [arXiv](https://arxiv.org/abs/2201.10190) | 블록 동기 디코딩으로 스트리밍 attention ASR |
| Conmer: Streaming Conformer without self-attention for interactive voice assistants | 2024 | [link](https://scholar.google.com/scholar?q=Conmer:+Streaming+Conformer+without+self-attention+for+interactive+voice+assistants) | self-attention 제거한 스트리밍 Conformer |
| Full-duplex Speech-to-text System for Estonian | 2024 | [link](https://scholar.google.com/scholar?q=Full-duplex+Speech-to-text+System+for+Estonian) | 동시 듣기·전사 풀듀플렉스 S2T(에스토니아어) |
| Stateful Conformer with Cache-based Inference for Streaming Automatic Speech Recognition | 2023 | [arXiv](https://arxiv.org/abs/2312.17279) | 캐시 기반 상태 유지 스트리밍 Conformer |

## Speech Recognition — Decoding & LM Integration
| Title | Year | Link | Summary |
|-------|------|------|---------|
| Speech recognition with weighted finite-state transducers | 2008 | [link](https://scholar.google.com/scholar?q=Speech+recognition+with+weighted+finite-state+transducers) | WFST로 사전·문법·음향 통합 디코딩 |
| End-to-end Speech Recognition with Word-based RNN Language Models | 2018 | [arXiv](https://arxiv.org/abs/1808.02608) | 단어 단위 RNN-LM 결합 디코딩 |
| Word Beam Search: A connectionist Temporal Classification Decoding Algorithm | 2018 | [link](https://scholar.google.com/scholar?q=Word+Beam+Search:+A+connectionist+Temporal+Classification+Decoding+Algorithm) | 사전 제약 CTC 빔서치 디코딩 |
| Component Fusion: Learning Replaceable Language Model Component for End-to-end Speech Recognition System | 2019 | [link](https://scholar.google.com/scholar?q=Component+Fusion:+Learning+Replaceable+Language+Model+Component+for+End-to-end+Speech+Recognition+System) | 교체 가능한 외부 LM 컴포넌트 융합 |
| Spell My Name: Keyword Boosted Speech Recognition | 2021 | [arXiv](https://arxiv.org/abs/2110.02791) | 키워드 문맥 부스팅으로 고유명사 개선 |
| Blank Collapse: Compressing CTC emission for the faster decoding | 2022 | [arXiv](https://arxiv.org/abs/2210.17017) | blank 프레임 축약으로 CTC 디코딩 가속 |

## Speech Recognition — Accented, Dialect & Robust
| Title | Year | Link | Summary |
|-------|------|------|---------|
| Using Accent-Specific Pronunciation Modelling for Robust Speech Recognition | 2003 | [link](https://scholar.google.com/scholar?q=Using+Accent-Specific+Pronunciation+Modelling+for+Robust+Speech+Recognition) | 억양별 발음 모델링 |
| MLLR-based Accent Model Adaptation without Accented Data | 2013 | [link](https://scholar.google.com/scholar?q=MLLR-based+Accent+Model+Adaptation+without+Accented+Data) | 억양 데이터 없이 MLLR 적응 |
| Adaptation Methods for Non-native Speech | 2014 | [link](https://scholar.google.com/scholar?q=Adaptation+Methods+for+Non-native+Speech) | 비원어민 음성 적응 기법 정리 |
| Introducing Attribute Features to Foreign Accent Recognition | 2016 | [link](https://scholar.google.com/scholar?q=Introducing+Attribute+Features+to+Foreign+Accent+Recognition) | 속성 피처로 외국 억양 인식 |
| Adversarial Multi-task Learning of Deep Neural Networks for Robust Speech Recognition | 2017 | [link](https://scholar.google.com/scholar?q=Adversarial+Multi-task+Learning+of+Deep+Neural+Networks+for+Robust+Speech+Recognition) | 노이즈 불변 위한 적대 멀티태스크 |
| Domain Adversarial Training for Accented Speech Recognition | 2018 | [arXiv](https://arxiv.org/abs/1806.02786) | 억양을 도메인으로 본 적대 학습 |
| End-to-end Accented Speech Recognition | 2018 | [link](https://scholar.google.com/scholar?q=End-to-end+Accented+Speech+Recognition) | 억양 음성 E2E 인식 |
| Improved Accented Speech Recognition using Accent Embeddings and Multi-task Learning | 2018 | [link](https://scholar.google.com/scholar?q=Improved+Accented+Speech+Recognition+using+Accent+Embeddings+and+Multi-task+Learning) | 억양 임베딩+멀티태스크 |
| Improving Noise Robustness of an End-to-End Neural Model for Automatic Speech Recognition | 2020 | [arXiv](https://arxiv.org/abs/2010.12715) | E2E ASR 잡음 강건성 향상 |
| Multi-dialect Speech Recognition with A Single Sequence-to-sequence Model | 2018 | [link](https://scholar.google.com/scholar?q=Multi-dialect+Speech+Recognition+with+A+Single+Sequence-to-sequence+Model) | 단일 seq2seq로 다방언 ASR |
| Adversarial Learning of Raw Speech Features for Domain Invariant Speech Recognition | 2019 | [link](https://scholar.google.com/scholar?q=Adversarial+Learning+of+Raw+Speech+Features+for+Domain+Invariant+Speech+Recognition) | 적대 학습으로 도메인 불변 raw 피처 |
| Coupled Training of Sequence-to-sequence Models for Accented Speech Recognition | 2019 | [link](https://scholar.google.com/scholar?q=Coupled+Training+of+Sequence-to-sequence+Models+for+Accented+Speech+Recognition) | 정상·억양 음성 결합 학습 |
| Leveraging Native Language Information for Improved Accented Speech Recognition | 2019 | [arXiv](https://arxiv.org/abs/1904.09038) | 모국어 정보 활용 억양 ASR |
| Automatic Speech Recognition of Multiple Accented Englsih Data | 2020 | [link](https://scholar.google.com/scholar?q=Automatic+Speech+Recognition+of+Multiple+Accented+Englsih+Data) | 다중 억양 영어 인식 연구 |
| Far-Field Automatic Speech Recognition | 2020 | [arXiv](https://arxiv.org/abs/2009.09395) | 원거리·잔향 환경 ASR 개관 |
| Multi-accent Speech Recognition with Hierarchical Grapheme Based Models | 2020 | [link](https://scholar.google.com/scholar?q=Multi-accent+Speech+Recognition+with+Hierarchical+Grapheme+Based+Models) | 계층적 grapheme 모델 다중 억양 |
| Speech Recognition of Multiple Accented English Data Using Acoustic Model Interpolation | 2020 | [link](https://scholar.google.com/scholar?q=Speech+Recognition+of+Multiple+Accented+English+Data+Using+Acoustic+Model+Interpolation) | 음향모델 보간으로 다중 억양 |
| Generalizing RNN-Transducer to Out-Domain Audio via Sparse Self-Attention Layers | 2021 | [arXiv](https://arxiv.org/abs/2108.10752) | sparse attention으로 도메인 외 일반화 |

## Speech Recognition — Efficiency & On-device
| Title | Year | Link | Summary |
|-------|------|------|---------|
| Streaming End-to-end Speech Recognition for Mobile Devices | 2018 | [arXiv](https://arxiv.org/abs/1811.06621) | 모바일용 RNN-T 스트리밍 ASR |
| Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling | 2023 | [arXiv](https://arxiv.org/abs/2311.00430) | 대규모 pseudo-label로 Whisper 증류 |
| DistillW2V2: A Small and Streaming Wav2vec 2.0 Based ASR Model | 2020 | [arXiv](https://arxiv.org/abs/2006.11477) | 소형·스트리밍 wav2vec2 증류 |
| Moonshine: Speech Recognition for Live Transcription and Voice Commands | 2024 | [arXiv](https://arxiv.org/abs/2410.15608) | 실시간 전사·명령용 경량 엣지 ASR |
| Edge-ASR: Towards Low-Bit Quantization of Automatic Speech Recognition Models | 2025 | [arXiv](https://arxiv.org/abs/2507.07877) | 엣지용 저비트 양자화 ASR |
| Pushing the Limits of On-Device Streaming ASR: A Compact, High-Accuracy English Model for Low-Latency Inference | 2025 | [link](https://scholar.google.com/scholar?q=Pushing+the+Limits+of+On-Device+Streaming+ASR:+A+Compact,+High-Accuracy+English+Model+for+Low-Latency+Inference) | 소형·고정확 온디바이스 스트리밍 ASR |

## Speech Recognition — Large-scale, Weak & Semi-supervised
| Title | Year | Link | Summary |
|-------|------|------|---------|
| Improved Noisy Student Training for Automatic Speech Recognition | 2020 | [arXiv](https://arxiv.org/abs/2005.09629) | 노이즈 학생 반복 학습 ASR 적용 |
| Iterative Pseudo-Labeling for Speech Recognition | 2020 | [arXiv](https://arxiv.org/abs/2005.09267) | 반복 가짜 라벨링 준지도 ASR |
| Pushing the Limits of Semi-Supervised Learning for Automatic Speech Recognition | 2020 | [arXiv](https://arxiv.org/abs/2010.10504) | wav2vec2+noisy student 대규모 준지도 |
| SpeechStew: Simply Mix All Available Speech Recognition Data to Train One Large Neural Network | 2021 | [arXiv](https://arxiv.org/abs/2104.02133) | 여러 코퍼스 단순 혼합 학습 |
| Robust Speech Recognition via Large-Scale Weak Supervision | 2022 | [arXiv](https://arxiv.org/abs/2212.04356) | 68만 시간 약지도 학습 Whisper |
| Enhancing Low-Resource ASR through Versatile TTS: Bridging the Data Gap | 2024 | [arXiv](https://arxiv.org/abs/2410.16726) | TTS 합성으로 저자원 ASR 데이터 보강 |
| Less is More: Accurate Speech Recognition & Translation without Web-Scale Data | 2024 | [arXiv](https://arxiv.org/abs/2406.19674) | 적은 고품질 데이터로 Canary ASR·AST |
| OWSM v3.1: Better and Faster Open Whisper-Style Speech Models based on E-Branchformer | 2024 | [arXiv](https://arxiv.org/abs/2401.16658) | 오픈 Whisper형 다국어·다태스크 모델 |

## Speech Recognition — Continual Learning
| Title | Year | Link | Summary |
|-------|------|------|---------|
| Continual Learning in Automatic Speech Recognition | 2021 | [link](https://scholar.google.com/scholar?q=Continual+Learning+in+Automatic+Speech+Recognition) | ASR 점진 학습의 망각 분석·기법 |
| Continual Learning Using Lattice-Free MMI for Speech Recognition | 2021 | [arXiv](https://arxiv.org/abs/2110.07055) | LF-MMI 기반 ASR 연속 학습 |
| Online Continual Learning of End-to-End Speech Recognition Models | 2022 | [arXiv](https://arxiv.org/abs/2207.05071) | 스트림 데이터로 E2E ASR 온라인 연속 학습 |
| Using Adapters to Overcome Catastrophic Forgetting in End-to-End Automatic Speech Recognition | 2022 | [arXiv](https://arxiv.org/abs/2203.16082) | 어댑터 추가로 ASR 망각 완화 |

## Speech Recognition — Multilingual
| Title | Year | Link | Summary |
|-------|------|------|---------|
| Adversarial Training for Multilingual Acoustic Modeling | 2019 | [arXiv](https://arxiv.org/abs/1906.07093) | 언어 불변 위한 적대 다국어 음향모델 |
| Bilingual End-to-End ASR with Byte-Level Subwords | 2022 | [arXiv](https://arxiv.org/abs/2205.00485) | byte-level subword로 이중언어 E2E ASR |
| Enhancing Neural Transducer for Multilingual ASR with Synchronized Language Diarization | 2023 | [link](https://scholar.google.com/scholar?q=Enhancing+Neural+Transducer+for+Multilingual+ASR+with+Synchronized+Language+Diarization) | 언어 분할 동기화로 다국어 transducer |
| Multilingual DistilWhisper: Efficient Distillation of Multi-task Speech Models via Language-Specific Experts | 2023 | [arXiv](https://arxiv.org/abs/2311.01070) | 언어별 전문가로 Whisper 증류 |
| Canary-1B-v2 & Parakeet-TDT-0.6B-v3: Efficient and High-Performance Models for Multilingual ASR and AST | 2025 | [arXiv](https://arxiv.org/abs/2509.14128) | 다국어 ASR·AST 고효율 NVIDIA 모델 |

## Speech Recognition — Code-switching
| Title | Year | Link | Summary |
|-------|------|------|---------|
| Exploring Lexicon-Free Modeling Units for End-to-End Korean and Korean-English Code-Switching Speech Recognition | 2019 | [arXiv](https://arxiv.org/abs/1910.11590) | 한·한영 CS용 lexicon-free 모델링 단위 비교 |
| Speech collage: code-switched audio generation by collaging monolingual corpora | 2023 | [arXiv](https://arxiv.org/abs/2309.15674) | 단일어 코퍼스 이어붙여 CS 오디오 생성 |
| Unified Model for Code-Switching Speech Recognition and Language Identification Based on Concatenated Tokenizer | 2023 | [arXiv](https://arxiv.org/abs/2306.08753) | 연결 토크나이저로 CS ASR+LID 통합 |
| Adapting Whisper for Code-Switching through Encoding Refining and Language-Aware Decoding | 2024 | [arXiv](https://arxiv.org/abs/2412.16507) | 인코딩 정제+언어 인식 디코딩으로 CS Whisper |
| Aligning Speech to Languages to Enhance Code-switching Speech Recognition | 2024 | [arXiv](https://arxiv.org/abs/2403.05887) | 음성-언어 정렬로 CS ASR 개선 |
| Code-Switching in End-to-End Automatic Speech Recognition: A Systematic Literature Review | 2024 | [link](https://scholar.google.com/scholar?q=Code-Switching+in+End-to-End+Automatic+Speech+Recognition:+A+Systematic+Literature+Review) | E2E CS ASR 체계적 문헌 리뷰 |
| Enhancing Code-switching Speech Recognition with Interactive Language Biases | 2023 | [arXiv](https://arxiv.org/abs/2309.16953) | 상호작용 언어 bias로 CS ASR 개선 |
| Improving Code-Switching Speech Recognition with TTS Data Augmentation | 2026 | [arXiv](https://arxiv.org/abs/2601.00935) | TTS 증강으로 CS ASR 데이터 보강 |
| AsyncSwitch: Asynchronous Text-Speech Adaptation for Code-Switched ASR | 2025 | [arXiv](https://arxiv.org/abs/2506.14190) | 텍스트-음성 비동기 적응 CS ASR |
| Boosting Code-Switching ASR with Mixture of Experts Enhanced Speech-Conditioned LLM | 2024 | [arXiv](https://arxiv.org/abs/2409.15905) | MoE 강화 speech-conditioned LLM CS ASR |
| Can we train ASR systems on Code-switch without real code-switch data? Case study for Singapore's languages | 2025 | [arXiv](https://arxiv.org/abs/2506.14177) | 실제 CS 데이터 없이 CS ASR 학습 가능성 |
| Code-switching Speech Recognition Under the Lens: Model- and Data-Centric Perspectives | 2025 | [arXiv](https://arxiv.org/abs/2509.24310) | 모델·데이터 관점 CS ASR 분석 |
| Gated Low-rank Adaptation for personalized Code-Switching Automatic Speech Recognition on the low-spec devices | 2024 | [arXiv](https://arxiv.org/abs/2406.02562) | 게이트 LoRA로 저사양 개인화 CS ASR |
| HiKE: Hierarchical Evaluation Framework for Korean-English Code-Switching Speech Recognition | 2025 | [arXiv](https://arxiv.org/abs/2509.24613) | 한영 CS ASR 계층적 평가 프레임워크 |

## Speech Recognition — General, Surveys & Features
| Title | Year | Link | Summary |
|-------|------|------|---------|
| An overview of Automatic Speech Attribute Transcription (ASAT) | 2009 | [link](https://scholar.google.com/scholar?q=An+overview+of+Automatic+Speech+Attribute+Transcription+(ASAT)) | 속성 기반 음성 전사 프레임워크 개요 |
| Japanese and Korean voice search | 2012 | [link](https://scholar.google.com/scholar?q=Japanese+and+Korean+voice+search) | 일·한 음성검색 시스템 구축 |
| A pitch extraction algorithm tuned for automatic speech recognition | 2014 | [link](https://scholar.google.com/scholar?q=A+pitch+extraction+algorithm+tuned+for+automatic+speech+recognition) | ASR용 Kaldi pitch 추출 알고리즘 |
| A Comparison of Sequence-to-Sequence Models for Speech Recognition | 2017 | [link](https://scholar.google.com/scholar?q=A+Comparison+of+Sequence-to-Sequence+Models+for+Speech+Recognition) | CTC/RNN-T/attention seq2seq 비교 |
| An exploration of dropout with LSTMs | 2017 | [link](https://scholar.google.com/scholar?q=An+exploration+of+dropout+with+LSTMs) | LSTM 음향모델 dropout 적용 탐구 |
| English Conversational Telephone Speech Recognition by Humans and Machines | 2017 | [arXiv](https://arxiv.org/abs/1703.02136) | SWB에서 인간 수준 도달 연구 |
| JHU Kaldi system for Arabic MGB-3 ASR challenge using diarization, audio-transcript alignment and transfer learning | 2017 | [link](https://scholar.google.com/scholar?q=JHU+Kaldi+system+for+Arabic+MGB-3+ASR+challenge+using+diarization,+audio-transcript+alignment+and+transfer+learning) | MGB-3 아랍어 챌린지 시스템 |
| Some Commonly Used Speech Feature Extraction Algorithms | 2018 | [link](https://scholar.google.com/scholar?q=Some+Commonly+Used+Speech+Feature+Extraction+Algorithms) | MFCC 등 음성 피처 추출 정리 |
| A Comparative Study on Transformer vs RNN in Speech Applications | 2019 | [link](https://scholar.google.com/scholar?q=A+Comparative+Study+on+Transformer+vs+RNN+in+Speech+Applications) | 음성 태스크에서 Transformer vs RNN 비교 |
| An overview of End-to-end Automatic Speech Recognition | 2019 | [link](https://scholar.google.com/scholar?q=An+overview+of+End-to-end+Automatic+Speech+Recognition) | E2E ASR 전반 서베이 |
| CTC-Segmentation of Large Corpora for German End-to-end Speech Recognition | 2020 | [arXiv](https://arxiv.org/abs/2007.09127) | CTC로 대규모 코퍼스 강제정렬·분할 |
| Multi-task Learning for Speech Recognition: An overview | 2020 | [link](https://scholar.google.com/scholar?q=Multi-task+Learning+for+Speech+Recognition:+An+overview) | ASR 멀티태스크 학습 개관 |
| 저자원 환경의 음성인식을 위한 자기 주의를 활용한 음향 모델 학습 | 2020 | [link](https://scholar.google.com/scholar?q=저자원+환경의+음성인식을+위한+자기+주의를+활용한+음향+모델+학습) | self-attention 활용 저자원 음향모델 학습 |
| Active Learning for LF-MMI Trained Neural Networks in ASR | 2021 | [link](https://scholar.google.com/scholar?q=Active+Learning+for+LF-MMI+Trained+Neural+Networks+in+ASR) | LF-MMI ASR의 액티브 러닝 |
| Hyperparameter experiments on end-to-end automatic speech recognition | 2021 | [link](https://scholar.google.com/scholar?q=Hyperparameter+experiments+on+end-to-end+automatic+speech+recognition) | E2E ASR 하이퍼파라미터 실험 |
| Active Learning for Speech Recognition: the Power of Gradients | 2016 | [arXiv](https://arxiv.org/abs/1612.03226) | gradient 크기로 라벨 샘플 선택 |

## Pronunciation Assessment & L2 Learning (CAPT)
| Title | Year | Link | Summary |
|-------|------|------|---------|
| Automatic Speech Recognition for Second Language Learning: How and Why It Actually Works | 2009 | [link](https://scholar.google.com/scholar?q=Automatic+Speech+Recognition+for+Second+Language+Learning:+How+and+Why+It+Actually+Works) | L2 학습용 ASR 적용 원리 |
| Exploring Deep Learning Architectures for Automatically Grading Non-native Spontaneous Speech | 2017 | [link](https://scholar.google.com/scholar?q=Exploring+Deep+Learning+Architectures+for+Automatically+Grading+Non-native+Spontaneous+Speech) | 비원어민 즉흥 발화 자동 채점 |
| A deep Learning Approach to Automatic Characterisation of Rhythm in Non-native English Speech | 2021 | [link](https://scholar.google.com/scholar?q=A+deep+Learning+Approach+to+Automatic+Characterisation+of+Rhythm+in+Non-native+English+Speech) | 딥러닝으로 비원어민 영어 리듬 특성화 |
| Computer-Assisted Pronunciation Training from Pronunciation Scoring Towards Spoken Language Learning | 2021 | [link](https://scholar.google.com/scholar?q=Computer-Assisted+Pronunciation+Training+from+Pronunciation+Scoring+Towards+Spoken+Language+Learning) | CAPT·발음 채점 서베이 |

## Speech LLM / Audio LLM
| Title | Year | Link | Summary |
|-------|------|------|---------|
| AudioGPT: Understanding and Generating Speech, Music, Sound, and Talking Head | 2023 | [arXiv](https://arxiv.org/abs/2304.12995) | LLM+오디오 전문가 도구 연결 |
| AudioPaLM: A Large Language Model That Can Speak and Listen | 2023 | [arXiv](https://arxiv.org/abs/2306.12925) | 텍스트+오디오 토큰 통합 LLM |
| Prompting Large Language Models with Speech Recognition Abilities | 2023 | [arXiv](https://arxiv.org/abs/2307.11795) | 오디오 임베딩 프롬프트로 LLM ASR 부여 |
| Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models | 2023 | [arXiv](https://arxiv.org/abs/2311.07919) | 다태스크 통합 대규모 오디오-언어모델 |
| SALM: Speech-augmented Language Model with In-context Learning for Speech Recognition and Translation | 2023 | [arXiv](https://arxiv.org/abs/2310.09424) | in-context 학습 음성 증강 LM |
| Speech Translation with Large Language Models: An Industrial Practice | 2023 | [arXiv](https://arxiv.org/abs/2312.13585) | LLM 음성번역 산업 적용 사례 |
| SpeechGPT: Empowering Large Language Models with Intrinsic Cross-Modal Conversational Abilities | 2023 | [arXiv](https://arxiv.org/abs/2305.11000) | 이산 음성 토큰으로 cross-modal 대화 LLM |
| Textually Pretrained Speech Language Models | 2023 | [arXiv](https://arxiv.org/abs/2305.13009) | 텍스트 LLM 초기화 음성 LM(TWIST) |
| Whispering LLaMA: A Cross-Modal Generative Error Correction Framework for Speech Recognition | 2023 | [arXiv](https://arxiv.org/abs/2310.06434) | LLM으로 ASR 생성 오류교정 |
| A Survey on Speech Large Language Models | 2024 | [link](https://scholar.google.com/scholar?q=A+Survey+on+Speech+Large+Language+Models) | 음성 LLM 전반 서베이 |
| Audio Flamingo: A Novel Audio Language Model with Few-Shot Learning and Dialogue Abilities | 2024 | [arXiv](https://arxiv.org/abs/2402.01831) | few-shot·대화형 오디오 언어모델 |
| BESTOW: Efficient and Streamable Speech Language Model with the Best of Two Worlds in GPT and T5 | 2024 | [arXiv](https://arxiv.org/abs/2406.19954) | GPT·T5 장점 결합 스트리밍 speech LM |
| GenTranslate: Large Language Models are Generative Multilingual Speech and Machine Translators | 2024 | [arXiv](https://arxiv.org/abs/2402.06894) | N-best를 LLM이 융합 번역 |
| GLM-4-Voice: Towards Intelligent and Human-Like End-to-End Spoken Chatbot | 2024 | [arXiv](https://arxiv.org/abs/2412.02612) | E2E 음성 챗봇 |
| Large Language Models are Efficient Learners of Noise-Robust Speech Recognition | 2024 | [arXiv](https://arxiv.org/abs/2401.10446) | LLM 기반 잡음 강건 생성 오류교정 |
| LLaMA-Omni: Seamless Speech Interaction with Large Language Models | 2024 | [arXiv](https://arxiv.org/abs/2409.06666) | 저지연 음성 대화 LLaMA |
| Mini-Omni: Language Models Can Hear, Talk While Thinking in Streaming | 2024 | [arXiv](https://arxiv.org/abs/2408.16725) | 실시간 듣고 말하는 음성 LLM |
| On decoder-only architecture for speech-to-text and large language model integration | 2023 | [arXiv](https://arxiv.org/abs/2307.03917) | decoder-only로 음성-LLM 통합(Speech-LLaMA) |
| Paralinguistics-Aware Speech-Empowered Large Language Models for Natural Conversation | 2024 | [arXiv](https://arxiv.org/abs/2402.05706) | 준언어 정보 인식 음성 대화 LLM(USDM) |
| SpeechAgents: Human-Communication Simulation with Multi-Modal Multi-Agent Systems | 2024 | [arXiv](https://arxiv.org/abs/2401.03945) | 멀티에이전트 인간 소통 시뮬레이션 |
| VoiceTextBlender: Augmenting Large Language Models with Speech Capabilities via Single-Stage Joint Speech-Text Supervised Fine-Tuning | 2024 | [arXiv](https://arxiv.org/abs/2410.17485) | 단일단계 공동 SFT로 음성 능력 부여 |
| Kimi-Audio Technical Report | 2025 | [arXiv](https://arxiv.org/abs/2504.18425) | 범용 오디오 이해·생성·대화 모델 |

## Speaker Recognition & Diarization
| Title | Year | Link | Summary |
|-------|------|------|---------|
| Deep speaker: an end-to-end neural speaker embedding system | 2017 | [arXiv](https://arxiv.org/abs/1705.02304) | E2E 화자 임베딩 시스템 |
| Generalized End-to-End Loss for Speaker Verifications | 2017 | [arXiv](https://arxiv.org/abs/1710.10467) | 배치 내 centroid 기반 손실(GE2E) |
| Speaker Recognition from Raw Waveform with SincNet | 2018 | [arXiv](https://arxiv.org/abs/1808.00158) | sinc 필터 학습 raw waveform 화자인식 |
| Unsupervised Domain Adaptation via Domain Adversarial Training for Speaker Recognition | 2018 | [link](https://scholar.google.com/scholar?q=Unsupervised+Domain+Adaptation+via+Domain+Adversarial+Training+for+Speaker+Recognition) | 적대 학습 비지도 도메인 적응 화자인식 |
| Deep Learning Methods in Speaker Recognition: A Review | 2019 | [link](https://scholar.google.com/scholar?q=Deep+Learning+Methods+in+Speaker+Recognition:+A+Review) | 딥러닝 화자인식 리뷰 |
| End-to-End Neural Speaker Diarization with Permutation-Free Objectives | 2019 | [arXiv](https://arxiv.org/abs/1909.05952) | 순열 불변 손실 E2E 화자분할(EEND) |
| End-to-End Neural Speaker Diarization with Self-Attention | 2019 | [arXiv](https://arxiv.org/abs/1909.06247) | self-attention EEND(SA-EEND) |
| RawNet: Advanced end-to-end deep neural network using raw waveforms for text-independent speaker verification | 2019 | [arXiv](https://arxiv.org/abs/1904.08104) | raw waveform E2E 화자검증 |
| AutoSpeech: Neural Architecture Search for Speaker Recognition | 2020 | [arXiv](https://arxiv.org/abs/2005.03215) | NAS로 화자인식 구조 탐색 |
| ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification | 2020 | [arXiv](https://arxiv.org/abs/2005.07143) | 채널 attention 강화 TDNN 화자검증 |
| Improved RawNet with Feature Map Scaling for Text-independent Speaker Verification using Raw Waveforms | 2020 | [arXiv](https://arxiv.org/abs/2004.00526) | feature map scaling 추가 RawNet |
| In defence of metric learning for speaker recognition | 2020 | [arXiv](https://arxiv.org/abs/2003.11982) | 화자인식 메트릭 러닝 손실 비교 |
| Speaker Recognition Based on Deep Learning: An Overview | 2020 | [arXiv](https://arxiv.org/abs/2012.00931) | 딥러닝 화자인식 개관 |
| End-to-end speaker segmentation for overlap-aware resegmentation | 2021 | [arXiv](https://arxiv.org/abs/2104.04045) | 중첩 인식 화자 세그멘테이션(pyannote) |
| TitaNet: Neural Model for Speaker Representation with 1D Depth-Wise Separable Convolutions and Global Context | 2021 | [arXiv](https://arxiv.org/abs/2110.04410) | 분리형 conv+전역문맥 화자 임베딩 |
| Multi-scale Speaker Diarization with Dynamic Scale Weighting | 2022 | [arXiv](https://arxiv.org/abs/2203.15974) | 다중 스케일 가중 화자분할(MSDD) |
| Pushing the limits of raw waveform speaker recognition | 2022 | [arXiv](https://arxiv.org/abs/2203.08488) | raw waveform 화자인식 한계 확장 |
| Speaker Targeting via Self-Speaker Adaptation for Multi-talker ASR | 2025 | [arXiv](https://arxiv.org/abs/2506.22646) | 자기 화자 적응으로 다화자 ASR 타깃팅 |

## Speech Enhancement
| Title | Year | Link | Summary |
|-------|------|------|---------|
| A Fully Convolutional Neural Network for Speech Enhancement | 2016 | [arXiv](https://arxiv.org/abs/1609.07132) | FCN 기반 음성 향상 |
| SEGAN: Speech Enhancement Generative Adversarial Network | 2017 | [arXiv](https://arxiv.org/abs/1703.09452) | GAN 기반 파형 음성향상 |
| Improved Speech Enhancement with the Wave-U-Net | 2018 | [arXiv](https://arxiv.org/abs/1811.11307) | 파형 도메인 U-Net 음성향상 |
| NARA-WPE: A Python package for weighted prediction error dereverberation in Numpy and Tensorflow for online and offline processing | 2018 | [link](https://scholar.google.com/scholar?q=NARA-WPE:+A+Python+package+for+weighted+prediction+error+dereverberation+in+Numpy+and+Tensorflow+for+online+and+offline+processing) | WPE 잔향제거 파이썬 패키지 |
| Pyroomacoustics: A Python Package for Audio Room Simulation and Array Processing Algorithms | 2018 | [arXiv](https://arxiv.org/abs/1710.04196) | 룸 시뮬레이션·어레이 처리 패키지 |
| Dual-Signal Transformation LSTM Network for Real-Time Noise Suppression | 2020 | [arXiv](https://arxiv.org/abs/2005.07551) | 실시간 잡음억제 DTLN |
| Real Time Speech Enhancement in the Waveform Domain | 2020 | [arXiv](https://arxiv.org/abs/2006.12847) | 실시간 파형 도메인 음성향상(Denoiser) |
| The INTERSPEECH 2020 Deep Noise Suppression Challenge: Datasets, Subjective Testing Framework, and Challenge Results | 2020 | [arXiv](https://arxiv.org/abs/2005.13981) | DNS 챌린지 데이터·평가·결과 |
| FAST-RIR: Fast neural diffuse room impulse response generator | 2021 | [arXiv](https://arxiv.org/abs/2110.04057) | 신경망 RIR 고속 생성기 |
| Self-Attentive VAD: Context-Aware Detection of Voice from Noise | 2021 | [link](https://scholar.google.com/scholar?q=Self-Attentive+VAD:+Context-Aware+Detection+of+Voice+from+Noise) | self-attention 문맥 인식 VAD |

## Speech Augmentation
| Title | Year | Link | Summary |
|-------|------|------|---------|
| Vocal Tract Length Perturbation (VTLP) improves speech recognition | 2013 | [link](https://scholar.google.com/scholar?q=Vocal+Tract+Length+Perturbation+(VTLP)+improves+speech+recognition) | 성도 길이 섭동 증강(VTLP) |
| Audio Augmentation for Speech Recognition | 2015 | [link](https://scholar.google.com/scholar?q=Audio+Augmentation+for+Speech+Recognition) | 속도 변형 등 오디오 증강 |
| Data Augmentation for Deep Neural Network Acoustic Modeling | 2015 | [link](https://scholar.google.com/scholar?q=Data+Augmentation+for+Deep+Neural+Network+Acoustic+Modeling) | DNN 음향모델용 데이터 증강 |
| A study on data augmentation of reverberant speech for robust speech recognition | 2017 | [link](https://scholar.google.com/scholar?q=A+study+on+data+augmentation+of+reverberant+speech+for+robust+speech+recognition) | 잔향 음성 증강으로 강건 ASR |
| Data Augmentation Improves Recognition of Foreign Accented Speech | 2018 | [link](https://scholar.google.com/scholar?q=Data+Augmentation+Improves+Recognition+of+Foreign+Accented+Speech) | 증강으로 외국 억양 인식 개선 |
| Specaugment on Large Scale Datasets | 2019 | [arXiv](https://arxiv.org/abs/1912.05533) | 대규모 데이터용 적응형 SpecAugment |
| SpecAugment: A simple Data Augmentation Method for Automatic Speech Recognition | 2019 | [arXiv](https://arxiv.org/abs/1904.08779) | 시간·주파수 마스킹 스펙트로그램 증강 |
| Speech Augmentation using Wavenet in Speech Recognition | 2019 | [link](https://scholar.google.com/scholar?q=Speech+Augmentation+using+Wavenet+in+Speech+Recognition) | WaveNet 합성 증강 |
| Data Augmenting Contrastive Learning of Speech Representations in the Time Domain | 2020 | [arXiv](https://arxiv.org/abs/2007.00991) | 시간영역 증강 contrastive(WavAugment) |
| MixSpeech: Data Augmentation for Low-Resource Automatic Speech Recognition | 2021 | [arXiv](https://arxiv.org/abs/2102.12664) | mixup 기반 저자원 ASR 증강 |

## Speech Toolkit
| Title | Year | Link | Summary |
|-------|------|------|---------|
| The Kaldi Speech Recognition Toolkit | 2011 | [link](https://scholar.google.com/scholar?q=The+Kaldi+Speech+Recognition+Toolkit) | WFST 기반 ASR 툴킷 Kaldi |
| Montreal Forced Aligner: trainable text-speech alignment using Kaldi | 2017 | [link](https://scholar.google.com/scholar?q=Montreal+Forced+Aligner:+trainable+text-speech+alignment+using+Kaldi) | 학습형 강제정렬 도구(MFA) |
| ESPnet: End-to-end Speech Processing Toolkit | 2018 | [arXiv](https://arxiv.org/abs/1804.00015) | E2E 음성처리 툴킷 |
| PyKaldi: A Python Wrapper for Kaldi | 2018 | [link](https://scholar.google.com/scholar?q=PyKaldi:+A+Python+Wrapper+for+Kaldi) | Kaldi 파이썬 래퍼 |
| Wav2Letter++: A Fast Open-source Speech Recognition System | 2018 | [arXiv](https://arxiv.org/abs/1812.07625) | C++ 고속 오픈소스 ASR |
| Espresso: A Fast End-to-End Neural Speech Recognition Toolkit | 2019 | [arXiv](https://arxiv.org/abs/1909.08723) | PyTorch 기반 E2E ASR 툴킷 |
| NeMo: a toolkit for building AI applications using Neural Modules | 2019 | [arXiv](https://arxiv.org/abs/1909.09577) | NVIDIA 음성·NLP 모듈 툴킷 |
| PyKaldi2: Yet Another Speech Toolkit Based on Kaldi and Pytorch | 2019 | [arXiv](https://arxiv.org/abs/1907.05955) | Kaldi+PyTorch 시퀀스 학습 툴킷 |
| The Pytorch-kaldi Speech Recognition Toolkit | 2018 | [arXiv](https://arxiv.org/abs/1811.07453) | Kaldi 피처+PyTorch 음향모델 |
| FAIRSEQ S2T: Fast Speech-to-Text Modeling with FAIRSEQ | 2020 | [arXiv](https://arxiv.org/abs/2010.05171) | fairseq 음성-텍스트 확장 |
| PyChain: A Fully Parallelized PyTorch Implementation of LF-MMI for End-to-End ASR | 2020 | [arXiv](https://arxiv.org/abs/2005.09824) | LF-MMI PyTorch 병렬 구현 |
| The 2020 ESPnet Update: New Features, Broadened Applications, Performance Improvements, and Future Plans | 2020 | [arXiv](https://arxiv.org/abs/2012.13006) | 2020 ESPnet 기능 업데이트 |
| ExKaldi-RT: A Real-Time Automatic Speech Recognition Extension Toolkit of Kaldi | 2021 | [link](https://scholar.google.com/scholar?q=ExKaldi-RT:+A+Real-Time+Automatic+Speech+Recognition+Extension+Toolkit+of+Kaldi) | Kaldi 실시간 ASR 확장 |
| Lhotse: a speech data representation library for the modern deep learning ecosystem | 2021 | [link](https://scholar.google.com/scholar?q=Lhotse:+a+speech+data+representation+library+for+the+modern+deep+learning+ecosystem) | 음성 데이터 표현·전처리 라이브러리 |
| Recent Developments on Espnet Toolkit Boosted By Conformer | 2020 | [arXiv](https://arxiv.org/abs/2010.13956) | Conformer 도입 ESPnet 업데이트 |
| SpeechBrain: A General-Purpose Speech Toolkit | 2021 | [arXiv](https://arxiv.org/abs/2106.04624) | 범용 PyTorch 음성 툴킷 |
| ESPnet-ONNX: Bridging a Gap Between Research and Production | 2022 | [link](https://scholar.google.com/scholar?q=ESPnet-ONNX:+Bridging+a+Gap+Between+Research+and+Production) | ESPnet 모델 ONNX 배포 |

## Speech Dataset
| Title | Year | Link | Summary |
|-------|------|------|---------|
| Librispeech: An ASR Corpus Based on Public Domain Audio Books | 2015 | [link](https://scholar.google.com/scholar?q=Librispeech:+An+ASR+Corpus+Based+on+Public+Domain+Audio+Books) | 1000시간 오디오북 영어 ASR 코퍼스 |
| MUSAN: A Music, Speech, and Noise Corpus | 2015 | [arXiv](https://arxiv.org/abs/1510.08484) | 음악·음성·잡음 코퍼스 |
| Common Voice: A Massively-Multilingual Speech Corpus | 2019 | [arXiv](https://arxiv.org/abs/1912.06670) | 크라우드소싱 다국어 음성 코퍼스 |
| Libri-Light: A Benchmark for ASR with Limited or No Supervision | 2019 | [arXiv](https://arxiv.org/abs/1912.07875) | 6만 시간 비/저지도 ASR 벤치마크 |
| ClovaCall: Korean Goal-Oriented Dialog Speech Corpus for Automatic Speech Recognition of Contact Centers | 2020 | [arXiv](https://arxiv.org/abs/2004.09367) | 한국어 콜센터 목적지향 대화 음성 코퍼스 |
| SD-QA: Spoken Dialectal Question Answering for the Real World | 2021 | [arXiv](https://arxiv.org/abs/2109.12072) | 방언 음성 QA 데이터셋 |
| FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech | 2022 | [arXiv](https://arxiv.org/abs/2205.12446) | 102개 언어 few-shot 평가셋 |
| WavCaps: A ChatGPT-Assisted Weakly-Labelled Audio Captioning Dataset for Audio-Language Multimodal Research | 2023 | [arXiv](https://arxiv.org/abs/2303.17395) | ChatGPT 보조 약지도 오디오 캡션 데이터셋 |
| CS-FLEURS: A Massively Multilingual and Code-Switched Speech Dataset | 2025 | [link](https://scholar.google.com/scholar?q=CS-FLEURS:+A+Massively+Multilingual+and+Code-Switched+Speech+Dataset) | 다국어 코드스위칭 음성 데이터셋 |

## Spoken Dialogue & Full-duplex (S2S)
| Title | Year | Link | Summary |
|-------|------|------|---------|
| Towards a Japanese Full-duplex Spoken Dialogue System | 2025 | [arXiv](https://arxiv.org/abs/2506.02979) | 일본어 풀듀플렉스 대화 시스템 |
| DuplexCascade: Full-Duplex Speech-to-Speech Dialogue with VAD-Free Cascaded ASR-LLM-TTS Pipeline and Micro-Turn Optimization | 2026 | [arXiv](https://arxiv.org/abs/2603.09180) | VAD 없는 캐스케이드 풀듀플렉스 S2S |
| Effects of Dialogue Corpora Properties on Fine-Tuning a Moshi-Based Spoken Dialogue Model | 2025 | [link](https://scholar.google.com/scholar?q=Effects+of+Dialogue+Corpora+Properties+on+Fine-Tuning+a+Moshi-Based+Spoken+Dialogue+Model) | 대화 코퍼스 특성이 Moshi 파인튜닝에 미치는 영향 |
| Incorporating Dialogue State Tracking into Japanese Full-duplex Task-oriented Spoken Dialogue Model | 2025 | [link](https://scholar.google.com/scholar?q=Incorporating+Dialogue+State+Tracking+into+Japanese+Full-duplex+Task-oriented+Spoken+Dialogue+Model) | DST 결합 일본어 풀듀플렉스 대화모델 |
| PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech Models | 2026 | [arXiv](https://arxiv.org/abs/2602.06053) | 음성·역할 제어 풀듀플렉스 대화모델 |
| SALM-Duplex: Efficient and Direct Duplex Modeling for Speech-to-Speech Language Model | 2025 | [arXiv](https://arxiv.org/abs/2505.15670) | 직접 듀플렉스 모델링 S2S LM |

## TTS / Voice Conversion
| Title | Year | Link | Summary |
|-------|------|------|---------|
| Wavenet: A Generative Model for Raw Audio | 2016 | [arXiv](https://arxiv.org/abs/1609.03499) | 자기회귀 raw 파형 생성 보코더 |
| Deep Voice: Real-time Neural Text-to-Speech | 2017 | [arXiv](https://arxiv.org/abs/1702.07825) | 실시간 신경망 TTS 파이프라인 |
| Natural tts synthesis by conditioning wavenet on mel spectrogram predictions | 2017 | [arXiv](https://arxiv.org/abs/1712.05884) | Tacotron 2 (mel+WaveNet) |
| Tacotron: Towards end-to-end speech synthesis | 2017 | [arXiv](https://arxiv.org/abs/1703.10135) | seq2seq E2E TTS 원조 |
| Adversarial Audio Synthesis | 2018 | [arXiv](https://arxiv.org/abs/1802.04208) | GAN 오디오 생성(WaveGAN) |
| Neural Speech Synthesis with Transformer Network | 2018 | [arXiv](https://arxiv.org/abs/1809.08895) | Transformer 기반 TTS |
| Waveglow: A flow-based generative network for speech synthesis | 2018 | [arXiv](https://arxiv.org/abs/1811.00002) | flow 기반 병렬 보코더 |
| AutoVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss | 2019 | [arXiv](https://arxiv.org/abs/1905.05879) | 오토인코더 병목 zero-shot 음성변환 |
| FastSpeech: Fast, Robust and Controllable Text to Speech | 2019 | [arXiv](https://arxiv.org/abs/1905.09263) | 비자기회귀 병렬 TTS |
| MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis | 2019 | [arXiv](https://arxiv.org/abs/1910.06711) | 경량 GAN 보코더 |
| Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search | 2020 | [arXiv](https://arxiv.org/abs/2005.11129) | flow+단조정렬 비자기회귀 TTS |
| HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis | 2020 | [arXiv](https://arxiv.org/abs/2010.05646) | 고품질 GAN 보코더 |
| VocGAN: A High-Fidelity Real-time Vocoder with a Hierarchically-nested Adversarial Network | 2020 | [arXiv](https://arxiv.org/abs/2007.15256) | 계층적 GAN 실시간 보코더 |
| A review of deep learning based speech synthesis | 2021 | [link](https://scholar.google.com/scholar?q=A+review+of+deep+learning+based+speech+synthesis) | 딥러닝 음성합성 리뷰 |
| A Survey on Neural Speech Synthesis | 2021 | [arXiv](https://arxiv.org/abs/2106.15561) | 신경망 TTS 종합 서베이 |
| Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech | 2021 | [arXiv](https://arxiv.org/abs/2106.06103) | VAE+flow+적대 E2E TTS(VITS) |
| Emotional Speech Synthesis With Rich And Granularized Control | 2021 | [link](https://scholar.google.com/scholar?q=Emotional+Speech+Synthesis+With+Rich+And+Granularized+Control) | 세밀한 감정 제어 TTS |
| YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion for Everyone | 2021 | [arXiv](https://arxiv.org/abs/2112.02418) | zero-shot 다화자 TTS·VC |
| Period VITS: Variational Inference with Explicit Pitch Modeling for End-To-End Emotional Speech Synthesis | 2022 | [arXiv](https://arxiv.org/abs/2210.15964) | 명시적 피치 모델링 감정 TTS |
| OpenVoice: Versatile Instant Voice Cloning | 2023 | [arXiv](https://arxiv.org/abs/2312.01479) | 즉시 음색 복제·스타일 제어 |
| VITS2: Improving Quality and Efficiency of Single-Stage Text-to-Speech with Adversarial Learning and Architecture Design | 2023 | [arXiv](https://arxiv.org/abs/2307.16430) | 구조 개선한 단일단계 TTS |

## NLP / LLM
| Title | Year | Link | Summary |
|-------|------|------|---------|
| An algorithm for suffix stripping | 1980 | [link](https://scholar.google.com/scholar?q=An+algorithm+for+suffix+stripping) | Porter 스테머 알고리즘 |
| Neural Machine Translation of Rare Words with Subword Units | 2015 | [arXiv](https://arxiv.org/abs/1508.07909) | BPE 서브워드 분절 |
| Attention is all you need | 2017 | [arXiv](https://arxiv.org/abs/1706.03762) | self-attention 기반 Transformer |
| BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | 2018 | [arXiv](https://arxiv.org/abs/1810.04805) | 양방향 MLM 사전학습 |
| SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing | 2018 | [arXiv](https://arxiv.org/abs/1808.06226) | 언어 독립 서브워드 토크나이저 |
| Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates | 2018 | [arXiv](https://arxiv.org/abs/1804.10959) | 다중 분절 샘플링 정규화 |
| BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension | 2019 | [arXiv](https://arxiv.org/abs/1910.13461) | 디노이징 seq2seq 사전학습 |
| FAIRSEQ: A Fast, Extensible Toolkit for Sequence Modeling | 2019 | [arXiv](https://arxiv.org/abs/1904.01038) | 시퀀스 모델링 툴킷 |
| Language Modeling with Deep Transformers | 2019 | [arXiv](https://arxiv.org/abs/1905.04226) | 깊은 Transformer 언어모델 |
| Language Models are Unsupervised Multitask Learners | 2019 | [link](https://scholar.google.com/scholar?q=Language+Models+are+Unsupervised+Multitask+Learners) | GPT-2, 비지도 멀티태스크 LM |
| Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism | 2019 | [arXiv](https://arxiv.org/abs/1909.08053) | 텐서 병렬로 초대형 LM 학습 |
| Neural Machine Translation with Byte-Level Subwords | 2019 | [arXiv](https://arxiv.org/abs/1909.03341) | 바이트 단위 서브워드(BBPE) |
| RoBERTa: A Robustly Optimized BERT Pretraining Approach | 2019 | [arXiv](https://arxiv.org/abs/1907.11692) | 학습 레시피 개선 BERT |
| The Curious Case of Neural Text Degeneration | 2019 | [arXiv](https://arxiv.org/abs/1904.09751) | nucleus(top-p) 샘플링 제안 |
| Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context | 2019 | [arXiv](https://arxiv.org/abs/1901.02860) | 세그먼트 순환 장문 LM |
| Big Bird: Transformers for Longer Sequences | 2020 | [arXiv](https://arxiv.org/abs/2007.14062) | 희소 attention 장문 Transformer |
| Electra: Pre-training text encoders as discriminators rather than generators | 2020 | [arXiv](https://arxiv.org/abs/2003.10555) | replaced token detection 사전학습 |
| Longformer: The Long-Document Transformer | 2020 | [arXiv](https://arxiv.org/abs/2004.05150) | 슬라이딩 윈도 attention 장문 |
| mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer | 2020 | [arXiv](https://arxiv.org/abs/2010.11934) | 다국어 T5 |
| Recent Trends in the Use of Deep Learning Models for Grammar Error Handling | 2020 | [arXiv](https://arxiv.org/abs/2009.02358) | 문법오류 처리 딥러닝 동향 |
| Transformers: State-of-the-Art Natural Language Processing | 2020 | [arXiv](https://arxiv.org/abs/1910.03771) | HuggingFace Transformers 라이브러리 |
| BTS: Back TranScription for Speech-to-Text Post-Processor using Text-to-Speech-to-Text | 2021 | [link](https://scholar.google.com/scholar?q=BTS:+Back+TranScription+for+Speech-to-Text+Post-Processor+using+Text-to-Speech-to-Text) | TTS-STT로 후처리 학습데이터 생성 |
| LoRA: Low-Rank Adaptation of Large Language Models | 2021 | [arXiv](https://arxiv.org/abs/2106.09685) | 저랭크 행렬 파라미터 효율 미세조정 |
| SimCSE: Simple Contrastive Learning of Sentence Embeddings | 2021 | [arXiv](https://arxiv.org/abs/2104.08821) | dropout 대조 문장 임베딩 |
| Chain-of-Thought Prompting Elicits Reasoning in Large Language Models | 2022 | [arXiv](https://arxiv.org/abs/2201.11903) | 단계적 추론 유도 프롬프팅 |
| Thutmose Tagger: Single-pass neural model for Inverse Text Normalization | 2022 | [arXiv](https://arxiv.org/abs/2208.00064) | 태깅 기반 역텍스트정규화(ITN) |

## Computer Vision
| Title | Year | Link | Summary |
|-------|------|------|---------|
| Rich feature hierarchies for accurate object detection and semantic segmentation | 2014 | [arXiv](https://arxiv.org/abs/1311.2524) | 영역 제안 기반 검출(R-CNN) |
| Very deep convolutional networks for large-scale image recognition | 2014 | [arXiv](https://arxiv.org/abs/1409.1556) | 3x3 conv 깊은 망(VGG) |
| Deep residual learning for image recognition | 2015 | [arXiv](https://arxiv.org/abs/1512.03385) | 잔차 연결 깊은 CNN(ResNet) |
| Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks | 2015 | [arXiv](https://arxiv.org/abs/1506.01497) | RPN 통합 객체검출 |
| U-Net: Convolutional Networks for Biomedical Image Segmentation | 2015 | [arXiv](https://arxiv.org/abs/1505.04597) | skip 연결 인코더-디코더 분할 |
| You Only Look Once: Unified, Real-Time Object detection | 2015 | [arXiv](https://arxiv.org/abs/1506.02640) | 단일 회귀 실시간 검출(YOLO) |
| Squeeze-and-excitation networks | 2017 | [arXiv](https://arxiv.org/abs/1709.01507) | 채널 재가중 모듈(SE) |
| EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks | 2019 | [arXiv](https://arxiv.org/abs/1905.11946) | 복합 스케일링 효율 CNN |
| Albumentations: fast and flexible image augmentations | 2018 | [arXiv](https://arxiv.org/abs/1809.06839) | 빠른 이미지 증강 라이브러리 |
| An image is worth 16x16 words: Transformers for image recognition at scale | 2020 | [arXiv](https://arxiv.org/abs/2010.11929) | 이미지 패치 Transformer(ViT) |

## Multimodal
| Title | Year | Link | Summary |
|-------|------|------|---------|
| Tensor Fusion Network for Multimodal Sentiment Analysis | 2017 | [arXiv](https://arxiv.org/abs/1707.07250) | 텐서곱 모달 융합 감성분석 |
| Multimodal Speech Emotion Recognition Using Audio and Text | 2018 | [arXiv](https://arxiv.org/abs/1810.04635) | 오디오+텍스트 듀얼 RNN 감정인식 |
| Multimodal Emotion Recognition with High-level Speech and Text Features | 2021 | [arXiv](https://arxiv.org/abs/2111.10202) | 고수준 음성·텍스트 융합 감정인식 |
| Multimodal Speech Emotion Recognition and Ambiguity Resolution | 2019 | [link](https://scholar.google.com/scholar?q=Multimodal+Speech+Emotion+Recognition+and+Ambiguity+Resolution) | 멀티모달 감정인식·모호성 해소 |
| Self-Supervised Learning with Cross-Modal Transformers for Emotion Recognition | 2020 | [arXiv](https://arxiv.org/abs/2011.10652) | cross-modal SSL 감정인식 |
| Multimodal Cross- and Self-Attention Network for Speech Emotion Recognition | 2021 | [link](https://scholar.google.com/scholar?q=Multimodal+Cross-+and+Self-Attention+Network+for+Speech+Emotion+Recognition) | cross/self-attention 음성감정인식 |
| Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation | 2022 | [arXiv](https://arxiv.org/abs/2211.06687) | 오디오-텍스트 대조 사전학습(CLAP) |
| Multimodal Speech Emotion Recognition using Cross Attention with Aligned Audio and Text | 2022 | [arXiv](https://arxiv.org/abs/2207.12895) | 정렬 cross-attention 감정인식 |
| Emotion-LLaMA: Multimodal Emotion Recognition and Reasoning with Instruction Tuning | 2024 | [arXiv](https://arxiv.org/abs/2406.11161) | 멀티모달 감정 인식·추론 LLM |
| Model Composition for Multimodal Large Language Models | 2024 | [link](https://scholar.google.com/scholar?q=Model+Composition+for+Multimodal+Large+Language+Models) | 모달 모델 조합으로 멀티모달 LLM |
| Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture-of-LoRAs | 2025 | [arXiv](https://arxiv.org/abs/2503.01743) | MoE-LoRA 소형 멀티모달 LM |
| Qwen2.5-Omni Technical Report | 2025 | [arXiv](https://arxiv.org/abs/2503.20215) | 전모달 입력·실시간 음성출력 모델 |

## Reinforcement Learning
| Title | Year | Link | Summary |
|-------|------|------|---------|
| Playing Atari with Deep Reinforcement Learning | 2013 | [arXiv](https://arxiv.org/abs/1312.5602) | 픽셀 입력 DQN |
| Neural architecture search with reinforcement learning | 2016 | [arXiv](https://arxiv.org/abs/1611.01578) | RL 컨트롤러로 구조 탐색(NAS) |

## Linguistics
| Title | Year | Link | Summary |
|-------|------|------|---------|
| Control Methods Used in a Study of the Vowels | 1952 | [link](https://scholar.google.com/scholar?q=Control+Methods+Used+in+a+Study+of+the+Vowels) | 모음 포먼트 고전 연구(Peterson & Barney) |
| On the distinction between 'stress-timed' and 'syllable-timed' languages | 1982 | [link](https://scholar.google.com/scholar?q=On+the+distinction+between+'stress-timed'+and+'syllable-timed'+languages) | 강세박자 vs 음절박자 구분 비판 |
| Stress-timing and Syllable-timing Reanalyzed | 1983 | [link](https://scholar.google.com/scholar?q=Stress-timing+and+Syllable-timing+Reanalyzed) | 박자 유형론 재분석(Dauer) |
| Language Discrimination by Newborns: Toward an Understanding of the Role of Rhythm | 1998 | [link](https://scholar.google.com/scholar?q=Language+Discrimination+by+Newborns:+Toward+an+Understanding+of+the+Role+of+Rhythm) | 신생아 리듬 기반 언어 구별 |
| Correlates of linguistic rhythm in the speech signal | 1999 | [link](https://scholar.google.com/scholar?q=Correlates+of+linguistic+rhythm+in+the+speech+signal) | 리듬 클래스 음향 상관(%V, ΔC) |
| Language identification with suprasegmental cues: A study based on speech resynthesis | 1999 | [link](https://scholar.google.com/scholar?q=Language+identification+with+suprasegmental+cues:+A+study+based+on+speech+resynthesis) | 초분절 단서 기반 언어 식별 |
| Accent as a Social Symbol | 2000 | [link](https://scholar.google.com/scholar?q=Accent+as+a+Social+Symbol) | 억양의 사회적 상징성 논의 |
| Durational Variability in Speech and the Rhythm Class Hypothesis | 2002 | [link](https://scholar.google.com/scholar?q=Durational+Variability+in+Speech+and+the+Rhythm+Class+Hypothesis) | PVI 지표 리듬 클래스 가설 |
| Calibrating rhythm: First language and second language studies | 2003 | [link](https://scholar.google.com/scholar?q=Calibrating+rhythm:+First+language+and+second+language+studies) | L1·L2 말리듬 측정 비교 |
| The Original ToBI System and the Evolution of the ToBI Framework | 2005 | [link](https://scholar.google.com/scholar?q=The+Original+ToBI+System+and+the+Evolution+of+the+ToBI+Framework) | 억양 라벨링 ToBI 체계 |
| Measures of Native and Non-Native Rhythm in a Quantity Language | 2008 | [link](https://scholar.google.com/scholar?q=Measures+of+Native+and+Non-Native+Rhythm+in+a+Quantity+Language) | 양적 언어의 원어민·비원어민 리듬 측정 |
| Voice Onset Time (VOT) at 50: Theoretical and practical issues in measuring voicing distinctions | 2018 | [link](https://scholar.google.com/scholar?q=Voice+Onset+Time+(VOT)+at+50:+Theoretical+and+practical+issues+in+measuring+voicing+distinctions) | VOT 측정 이론·실무 회고 |
| History of ESL Pronunciation Teaching | — | [link](https://scholar.google.com/scholar?q=History+of+ESL+Pronunciation+Teaching) | ESL 발음 교육사 |
| Intonation | — | [link](https://scholar.google.com/scholar?q=Intonation) | 억양(인토네이션) 개론 |
| On the Historical Phonotactic of English | — | [link](https://scholar.google.com/scholar?q=On+the+Historical+Phonotactic+of+English) | 영어 음소배열 역사 |
| Relations between language rhythm and speech rate | — | [link](https://scholar.google.com/scholar?q=Relations+between+language+rhythm+and+speech+rate) | 말리듬과 발화속도 관계 |
| Sound Change And Syllable Structure in Germanic Phonology | — | [link](https://scholar.google.com/scholar?q=Sound+Change+And+Syllable+Structure+in+Germanic+Phonology) | 게르만어 음변화·음절구조 |
| Speech rhythm across turn transitions in cross-cultural talk-in-interaction | — | [link](https://scholar.google.com/scholar?q=Speech+rhythm+across+turn+transitions+in+cross-cultural+talk-in-interaction) | 대화 턴 전환의 말리듬 |
| The Environment for Open-syllable Lengthening in Middle English | — | [link](https://scholar.google.com/scholar?q=The+Environment+for+Open-syllable+Lengthening+in+Middle+English) | 중세영어 개음절 장음화 환경 |
| The Historical Evolution of English Pronunciation | — | [link](https://scholar.google.com/scholar?q=The+Historical+Evolution+of+English+Pronunciation) | 영어 발음의 역사적 변천 |
| The Past, Present and Future of English Rhythm | — | [link](https://scholar.google.com/scholar?q=The+Past,+Present+and+Future+of+English+Rhythm) | 영어 리듬 연구 회고·전망 |
| 그림의 법칙: 연쇄 밀기 입장과 연쇄 당김 입장 | — | [link](https://scholar.google.com/scholar?q=그림의+법칙:+연쇄+밀기+입장과+연쇄+당김+입장) | 그림의 법칙: 연쇄 밀기 vs 당김 관점 |

