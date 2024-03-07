python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test1' --audio_feature='IS09_FRA' --text_feature='MARBERTv2-4-FRA' --video_feature='resnet50face_FRA' --lr=1e-3 --gpu=0

python -u main-release.py --dataset='MER2023' --model_type='mult' --feat_type='frm_align' --test_sets='test1' --audio_feature='IS09_FRA' --text_feature='MARBERTv2-4-FRA' --video_feature='resnet50face_FRA' --model='mult' --lr=1e-3 --gpu=0

python -u main-release.py --dataset='MER2023' --model_type='tfn' --feat_type='frm_align' --test_sets='test1' --audio_feature='IS09_FRA' --text_feature='MARBERTv2-4-FRA' --video_feature='resnet50face_FRA' --model='tfn' --lr=1e-3 --gpu=0

python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test1' --audio_feature='chinese-hubert-large-FRA' --text_feature='Baichuan-13B-Base-FRA' --video_feature='clip-vit-large-patch14-FRA' --lr=1e-3 --gpu=0

python -u main-release.py --dataset='MER2023' --model_type='mult' --feat_type='frm_align' --test_sets='test1' --audio_feature='IS09_FRA' --text_feature='MARBERTv2-4-FRA' --video_feature='resnet50face_FRA' --model='mult' --lr=1e-3 --gpu=0

python -u main-release.py --dataset='MER2023' --model_type='tfn' --feat_type='frm_align' --test_sets='test1' --audio_feature='IS09_FRA' --text_feature='MARBERTv2-4-FRA' --video_feature='resnet50face_FRA' --model='tfn' --lr=1e-3 --gpu=0

python -u main-release.py --dataset='MER2023' --model_type='attention' --test_sets='test' --audio_feature='chinese-hubert-large-FRA' --text_feature='Baichuan-13B-Base-FRA' --video_feature='clip-vit-large-patch14-FRA' --model='attention' --lr=1e-3 --gpu=0 --feat_type='frm_align'

====== Gain predition on test data =======
save results in ./saved-trimodal/model/cv_features:chinese-hubert-large-FRA+Baichuan-13B-Base-FRA+clip-vit-large-patch14-FRA_f1:0.0000_val:2.7753_metric:2.7753_1707678379.9435723.npz
save results in ./saved-trimodal/model/test_features:chinese-hubert-large-FRA+Baichuan-13B-Base-FRA+clip-vit-large-patch14-FRA_nores_1707678379.9435723.npz

CMUMOSI Attention
save results in ./saved-trimodal/result/cv_features:Baichuan-13B-Base-FRA+chinese-hubert-large-FRA+clip-vit-large-patch14-FRA_dataset:CMUMOSI_model:attention+frm_align+None_f1:0.6421_acc:0.6435_1707733511.0893695.npz
save results in ./saved-trimodal/result/test1_features:Baichuan-13B-Base-FRA+chinese-hubert-large-FRA+clip-vit-large-patch14-FRA_dataset:CMUMOSI_model:attention+frm_align+None_f1:0.6555_acc:0.6601_1707733511.0893695.npz
CMUMOSI MULT
f1:0.8336_acc:0.8333_1707735735.528314.npz
save results in ./saved-trimodal/result/test1_features:Baichuan-13B-Base-FRA+chinese-hubert-large-FRA+clip-vit-large-patch14-FRA_dataset:CMUMOSI_model:mult+frm_align+None_f1:0.7796_acc:0.7790_1707735735.528314.npz
CMUMOSI MFN
0.7855_acc:0.7870_1707738373.0271626.npz
save results in ./saved-trimodal/result/test1_features:Baichuan-13B-Base-FRA+chinese-hubert-large-FRA+clip-vit-large-patch14-FRA_dataset:CMUMOSI_model:mfn+frm_align+None_f1:0.6916_acc:0.6905_1707738373.0271626.npz
CMUMOSI MFM
f1:0.4388_acc:0.5556_1707739436.4398181.npz
save results in ./saved-trimodal/result/test1_features:Baichuan-13B-Base-FRA+chinese-hubert-large-FRA+clip-vit-large-patch14-FRA_dataset:CMUMOSI_model:mfm+frm_align+None_f1:0.3557_acc:0.4558_1707739436.4398181.npz
