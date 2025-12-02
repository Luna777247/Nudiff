# HOVER-NET: PHÂN ĐOẠN VÀ PHÂN LOẠI ĐỒNG THỜI CÁC HẠT NHÂN TRONG ẢNH MÔ BỆNH HỌC ĐA MÔ

Link: https://www.researchgate.net/publication/335888158_HoVer-Net_Simultaneous_Segmentation_and_Classification_of_Nuclei_in_Multi-Tissue_Histology_Images

**Tác giả**: Simon Graham¹, Quoc Dang Vu¹, Shan E Ahmed Raza¹, Yee-Wah Tsang², Jin Tae Kwak³, Nasir Rajpoot¹ (tác giả liên hệ)  
¹ Tissue Image Analytics Laboratory, Department of Computer Science, University of Warwick, UK  
² Department of Pathology, University Hospitals Coventry & Warwickshire, UK  
³ School of Electrical and Electronics Engineering, Chung-Ang University, Seoul, Hàn Quốc  

**Xuất bản**: Medical Image Analysis, Volume 58, December 2019, 101563  
**DOI**: 10.1016/j.media.2019.101563  
**arXiv preprint**: Không có (bản chính thức trên tạp chí)  
**Code chính thức**: https://github.com/simongraham/hover_net (và nhiều fork khác)

### Tóm tắt (Abstract)

Phân đoạn chính xác và phân loại hạt nhân trong ảnh mô bệnh học H&E là rất quan trọng để trích xuất thông tin lâm sàng từ toàn bộ slide kỹ thuật số (WSI).  

Mặc dù các phương pháp deep learning gần đây đã có tiến bộ vượt bậc, việc tách rời các hạt nhân chạm nhau vẫn là thách thức lớn do hiện tượng over-segmentation/under-segmentation của các kỹ thuật post-processing truyền thống (watershed, distance transform...).  

Chúng tôi đề xuất **HoVer-Net** – một mạng convolution end-to-end thực hiện **đồng thời instance segmentation và classification** của hạt nhân.  

Kiến trúc chính gồm 3 decoder branches:

- **NP branch** (Nuclear Pixel): semantic segmentation (pixel nào thuộc hạt nhân).  
- **HoVer branch** (Horizontal & Vertical): hồi quy (regression) hai bản đồ khoảng cách ngang (H) và dọc (V) từ mỗi pixel đến **tâm khối lượng (centroid)** của chính hạt nhân đó.  
- **NC branch** (Nuclear Classification): phân loại 5 loại hạt nhân (neoplastic, inflammatory, connective, dead, non-neoplastic epithelial).

HoVer map vượt trội hơn distance transform thông thường vì **robust với shape bất kỳ và các hạt nhân chạm nhau** (không bị ảnh hưởng bởi hình dạng bất quy tắc).  

Post-processing đơn giản (gradient ascent + grouping) để thu instance masks từ HoVer map.  

HoVer-Net đạt **state-of-the-art** trên MoNuSeg (MoNuSAC challenge), Kumar dataset và đặc biệt trên **CoNSeP** – dataset mới do nhóm tác giả công bố (41 WSI ung thư đại trực tràng, có cả segmentation + classification labels).  

Code được công khai và vẫn là baseline mạnh nhất cho nuclei segmentation đến tận 2024–2025.

### 1. Giới thiệu

Phân loại hạt nhân (nuclear pleomorphism, mitotic activity, lymphocyte infiltration...) là yếu tố quan trọng trong grading ung thư.  

Ví dụ: số lượng inflammatory nuclei liên quan đến phản ứng miễn dịch và tiên lượng tốt hơn ở một số loại ung thư.  

Thách thức lớn:

- Hạt nhân đa dạng kích thước, hình dạng, intensity, chồng chéo rất nhiều, nhuộm không đều.  
Các phương pháp cũ (watershed + hand-crafted features) dễ bị lỗi khi hạt nhân chạm nhau.  

Deep learning gần đây:

- DCAN, Micro-Net, DIST, StarDist, CIA-Net, Mask R-CNN... vẫn gặp khó khăn với clustered nuclei.  
- Phân loại hạt nhân thường làm riêng (classification sau segmentation) → lỗi lan truyền (error propagation).

HoVer-Net giải quyết bằng cách **học đồng thời segmentation và classification end-to-end**, dùng HoVer map mới để tách instance chính xác hơn.

### 2. Phương pháp chi tiết

#### 2.1 HoVer Map (cách biểu diễn instance mới – đóng góp chính)

Thay vì dùng distance transform (khoảng cách Euclidean đến biên) hoặc star-convex (StarDist), tác giả đề xuất bản **HoVer map**:

- Với mỗi hạt nhân instance, tạo 2 kênh:

  - Kênh H: Δx = x_pixel − x_centroid  
  - Kênh V: Δy = y_pixel − y_centroid

- Pixel không thuộc hạt nhân nào → (0,0).

Ưu điểm:

- Các pixel cùng một hạt nhân có giá trị gần giống nhau → dễ nhóm lại dù shape bất quy tắc.  
- Robust hơn distance transform (khi hạt nhân chạm nhau, distance transform dễ bị lẫn, còn HoVer thì không).  
- Không cần post-processing phức tạp như marker-controlled watershed.

#### 2.2 Kiến trúc mạng

- Backbone: ResNet-50 (pre-trained trên ImageNet) hoặc DenseNet-121.  
- Sau backbone: 3 decoder branches song song (mỗi branch 4 khối conv 3×3 + upsampling):

  - NP: output 2 lớp (background + foreground) → Dice loss.  
  - HoVer: output 2 kênh (H,V) → MSE loss (chỉ tính trên foreground).  
  - NC: output 5+1 lớp (5 loại + background) → weighted cross-entropy (tập trung vào vùng gần tâm hạt nhân).

Huấn luyện end-to-end, inference tốc độ ~0.19s/patch 1000×1000 trên GPU.

#### 2.3 Post-processing

- Từ HoVer map dự đoán → dùng gradient ascent để tìm local maxima (tâm hạt nhân).  
- Gán mỗi pixel cho tâm gần nhất theo khoảng cách Euclidean trong không gian HoVer.  
- Lọc nhiễu bằng threshold trên NP map.

### 3. Dataset & Kết quả

- **MoNuSeg** (44 ảnh train/test, multi-organ): HoVer-Net đạt **PQ = 0.614** (vô địch MoNuSAC challenge 2020).  
- **Kumar** (30 ảnh, multi-organ): F1 segmentation cao hơn các phương pháp khác.  
- **CoNSeP** (41 WSI colorectal, 24k hạt nhân có label loại):  

  - Segmentation: PQ = 0.672  
  - Classification: multi-class F1 = 0.782 (neoplastic & connective khó nhất)

Ablation study:

- HoVer map > distance transform > direct centroid offset regression.  
- 3 branches cùng train giúp tăng hiệu suất cả segmentation lẫn classification so với train riêng.

So sánh với SOTA tại thời điểm (2019): vượt DCAN, Micro-Net, Mask R-CNN, DIST...

### Kết luận chính của bài báo

- HoVer-Net là mô hình đầu tiên làm **đồng thời segmentation và 5-class classification** end-to-end với hiệu suất vượt trội.  
- HoVer map là đóng góp kỹ thuật quan trọng nhất – trở thành chuẩn mực mới (được dùng trong hầu hết paper sau này: HoVer-Net+, StarDist, NeNuSeg, TransNuSe, etc.).  
- Giới thiệu dataset CoNSeP (vẫn được dùng rộng rãi đến nay).  
- Code được cộng đồng duy trì và cải tiến, hiện tại là baseline mạnh nhất trên PanNuke dataset (2020, cũng do nhóm Warwick phát hành).

Đây là **paper kinh điển** trong pathology AI năm 2019 – gần như mọi paper nuclei segmentation sau này đều so sánh với HoVer-Net. Đến 2025, nó vẫn là một trong những baseline mạnh nhất (cùng với StarDist, CellViT, etc.).