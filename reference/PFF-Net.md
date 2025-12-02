# PANOPTIC FEATURE FUSION NET: MỘT PARADIGM INSTANCE SEGMENTATION MỚI CHO ẢNH Y SINH VÀ SINH HỌC

**Tác giả**: Dongnan Liu, Donghao Zhang, Yang Song, Heng Huang, Weidong Cai  
**Đơn vị**: The University of Sydney, UNSW Sydney, University of Pittsburgh  
**Xuất bản**: IEEE Transactions on Image Processing, Vol. 30, pp. 2045–2058, 2021  
**DOI**: 10.1109/TIP.2021.3050668  

### Tóm tắt (Abstract)

Instance segmentation là nhiệm vụ cực kỳ quan trọng trong phân tích ảnh y sinh và sinh học, đòi hỏi vừa phân loại từng pixel vừa tách riêng từng object cùng lớp.  

Tuy nhiên, các thách thức lớn vẫn tồn tại:  
- Nền phức tạp (cytoplasm, stroma…) trông giống foreground.  
- Biến đổi rất lớn về kích thước, hình dạng, texture, intensity của các object.  
- Các object chồng chéo nhau, biên mờ do nhuộm không đều.  

Các phương pháp deep learning hiện nay chia thành hai loại:  
- **Proposal-free**: dự đoán semantic segmentation trước → post-processing tách instance → dễ tạo biên nhân tạo khi object chồng chéo.  
- **Proposal-based** (Mask R-CNN…): detect bounding box trước → segment mask trong box → tách tốt object chồng chéo nhưng thiếu global semantic context.  

Cả hai loại đều bị **mất mát thông tin** vì chỉ tập trung hoặc global semantic hoặc local instance.

**PFFNet (Panoptic Feature Fusion Net)** – đề xuất trong bài báo này – là mở rộng của Cell R-CNN V2 (IJCAI 2020), thống nhất semantic và instance features trong một mạng duy nhất, kế thừa tinh thần panoptic segmentation nhưng tối ưu cho ảnh y sinh/sinh học.

**Các cải tiến chính so với Cell R-CNN V2**:

1. **Residual Attention Feature Fusion (RAFF)**  
   Thay vì thay thế trực tiếp feature từ instance branch vào semantic branch, RAFF dùng cơ chế residual + attention để **giữ toàn bộ semantic context** trong khi bổ sung instance-level details → semantic branch học được thông tin instance tốt hơn.

2. **Semantic Consistency Regularization**  
   Có hai semantic segmentation heads (một ở semantic branch, một ở instance branch). Thêm loss consistency (L1 hoặc cosine) để buộc hai đầu ra semantic phải giống nhau → học robust hơn, tránh overfitting.

3. **Mask Quality Sub-branch**  
   Dự đoán thêm một quality score (dựa trên Dice/IoU so với ground-truth trong training) cho mỗi mask.  
   Trong inference: final_score = classification_score × quality_score → loại bỏ các mask kém chất lượng dù classification score cao (vấn đề phổ biến của Mask R-CNN).

Kết quả: PFFNet vượt xa SOTA trên nhiều dataset:

| Dataset                  | Phương pháp tốt nhất trước | PFFNet (AP) |
|--------------------------|----------------------------|-------------|
| 2018 Data Science Bowl (nuclei) | 0.582 (Mask R-CNN V2)      | **0.682**   |
| MoNuSeg (nuclei)         | ~0.69 (DCAN, Hover-Net…)   | **0.743**   |
| BBBC006 (microscopy)     | –                          | **0.912**   |
| Plant Phenotyping (leaves) | –                        | **0.821**   |

→ cải thiện rất lớn (5–10 điểm AP) trên mọi loại ảnh y sinh/sinh học.

### 1. Giới thiệu (Introduction)

Instance segmentation là bước tiền xử lý bắt buộc để nghiên cứu hình thái, vị trí không gian, phân bố của các đối tượng sinh học (hạt nhân ung thư, lá cây, tế bào huỳnh quang…).

Thủ công thì chậm, không tái lập được do biến thiên giữa các bác sĩ.

Các phương pháp truyền thống dựa threshold hoặc handcrafted feature không hiệu quả kém vì các vấn đề đã nêu ở trên.

Deep learning hiện nay chia hai nhánh chính: proposal-free và proposal-based, đều có nhược điểm mất thông tin toàn cục hoặc chi tiết cục bộ.

Panoptic segmentation (Kirillov et al., CVPR 2019) và Panoptic FPN (2020) đã cố gắng kết hợp hai nhánh, nhưng vẫn train riêng hoặc chỉ share backbone → chưa tối ưu.

Tác giả trước đó đã đề xuất Cell R-CNN (MICCAI 2019) và Cell R-CNN V2 (IJCAI 2020) là những công trình đầu tiên áp dụng ý tưởng panoptic vào nuclei segmentation.

**PFFNet (Cell R-CNN V3)** cải thiện thêm ba điểm chính nêu trên để khắc phục hạn chế còn lại của V2.

### 2. Phương pháp (PFFNet Architecture)

Dựa trên Mask R-CNN + FPN backbone (ResNet-50/101).

Có hai branch chính:

- **Semantic branch**: dự đoán semantic segmentation (foreground vs background + optional multi-class).  
- **Instance branch**: Mask R-CNN head (bbox detection + mask head).

**Các thành phần mới**:

1. **Residual Attention Feature Fusion (RAFF)**  
   Lấy mask prediction từ instance branch → resize về kích thước feature map của semantic branch → dùng sigmoid làm attention map → nhân với feature semantic → cộng residual với feature gốc semantic → đầu ra semantic branch giàu instance information hơn.

2. **Semantic Head trong Instance Branch**  
   Từ feature của RoIAlign, thêm một head dự đoán semantic map cho toàn ảnh (không chỉ trong bbox).  
   → Tổng loss semantic = loss_semantic_branch + loss_semantic_from_instance + consistency_loss(L1 giữa hai đầu ra).

3. **Mask Quality Sub-branch**  
   Song song với mask head, thêm một fully-connected layer dự đoán quality score (0–1) dựa trên IoU/Dice với ground-truth trong training.  
   Loss = Binary Cross Entropy với target = IoU(mask_pred, GT).  
   Inference: final confidence = cls_score × mask_quality_score.

Loss tổng = Loss_detection + Loss_mask + Loss_semantic_main + Loss_semantic_instance + Loss_consistency + Loss_quality.

### 3. Thí nghiệm & Kết quả

Dataset:

- 2018 Data Science Bowl (nuclei biến đổi lớn)  
- MoNuSeg (multi-organ nuclei)  
- BBBC006 (fluorescence microscopy)  
- Plant phenotyping dataset (lá cây Arabidopsis)

PFFNet vượt tất cả SOTA tại thời điểm 2021 (Mask R-CNN, PAN, DCAN, Hover-Net, CIA-Net, Cell R-CNN V1/V2…).

Cải thiện đặc biệt lớn ở các trường hợp object chồng chéo và biên mờ.

Ablation study chứng minh từng thành phần mới đều mang lại cải thiện đáng kể.

### Kết luận chính của bài báo

- PFFNet là công trình đầu tiên thực sự **hòa trộn sâu** semantic và instance features trong một mạng duy nhất cho ảnh y sinh/sinh học.  
- Ba cải tiến RAFF + consistency + mask quality giúp vượt trội SOTA rất xa trên nhiều dataset khác nhau.  
- Mở ra hướng panoptic segmentation cho biomedical imaging (sau này ảnh hưởng đến rất nhiều paper nuclei segmentation 2021–2023 như TransNuSS, HoVer-Net+, StarDist panoptic version…).

Đây là một trong những paper kinh điển về instance segmentation y sinh thời điểm 2021, cùng thời với Hover-Net và trước khi diffusion/lớn model chiếm lĩnh. PFFNet vẫn là baseline mạnh trên MoNuSeg đến tận 2023–2024.