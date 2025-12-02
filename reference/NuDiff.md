# Tăng Cường Dữ Liệu Dựa Trên Mô Hình Khuếch Tán cho Phân Đoạn Ảnh Nhân Tế Bào

## Tác giả
X. Yu và cộng sự

---

### Quá trình khuếch tán ngược
**Quá trình khuếch tán thuận** → **Cấu trúc nhân tế bào**
![alt text](<../images/hinh 1.png>)
**Hình 1:** Minh họa mô hình khuếch tán trong ngữ cảnh cấu trúc nhân tế bào.

---

## 1. Giới Thiệu

Phân đoạn nhân tế bào là một bước cơ bản trong phân tích ảnh y học. Việc phân đoạn chính xác các nhân tế bào giúp phân tích hình ảnh mô bệnh học để hỗ trợ chẩn đoán và tiên lượng lâm sàng. Trong những năm gần đây, nhiều phương pháp phân đoạn nhân tế bào dựa trên học sâu đã được đề xuất [5, 18, 19, 23]. Hầu hết các phương pháp này được giám sát hoàn toàn nên hiệu suất phân đoạn tốt thường phụ thuộc vào một lượng lớn ảnh được gán nhãn. Tuy nhiên, việc gán nhãn thủ công các pixel thuộc về tất cả các ranh giới nhân trong một ảnh rất tốn thời gian và đòi hỏi kiến thức chuyên môn. Trong thực tế, rất khó để có được một lượng lớn ảnh mô bệnh học với chú thích mức pixel dày đặc nhưng có thể thu thập được một vài ảnh được gán nhãn. Một câu hỏi được đặt ra một cách tự nhiên: liệu chúng ta có thể mở rộng tập dữ liệu huấn luyện với một tỷ lệ nhỏ ảnh được gán nhãn để đạt hoặc thậm chí vượt qua hiệu suất phân đoạn của baseline được giám sát hoàn toàn hay không? Một cách trực quan, vì các ảnh được gán nhãn là các mẫu từ tổng thể các ảnh mô bệnh học, nếu phân phối cơ bản của các ảnh mô bệnh học được học, người ta có thể tạo ra vô số ảnh và nhãn ở mức pixel của chúng để tăng cường tập dữ liệu gốc. Do đó, cần phải phát triển một công cụ có khả năng học phân phối và tạo ra các cặp mẫu mới cho phân đoạn.

Mạng đối sinh (GANs) [2, 4, 12, 16, 20] đã được sử dụng rộng rãi trong tăng cường dữ liệu [11, 22, 27, 31]. Đặc biệt, một phương pháp dựa trên GAN mới được đề xuất có thể tổng hợp ảnh mô bệnh học được gán nhãn cho phân đoạn nhân tế bào [21]. Mặc dù GANs có khả năng tạo ra hình ảnh chất lượng cao, chúng nổi tiếng với việc huấn luyện không ổn định và thiếu sự đa dạng trong quá trình tạo sinh do chiến lược huấn luyện đối kháng. Gần đây, các mô hình khuếch tán được đại diện bởi mô hình xác suất khuếch tán khử nhiễu (DDPM) [8] có xu hướng vượt trội hơn GANs. Do cơ sở lý thuyết và hiệu suất ấn tượng của các mô hình khuếch tán, chúng đã sớm được áp dụng cho nhiều tác vụ thị giác, chẳng hạn như lấp đầy ảnh, siêu phân giải [30], dịch văn bản thành ảnh, phát hiện bất thường và phân đoạn [1, 9, 24, 26]. Là các mô hình dựa trên hàm hợp lý, các mô hình khuếch tán không yêu cầu huấn luyện đối kháng và vượt trội hơn GANs về sự đa dạng của các hình ảnh được tạo ra [3], điều này tự nhiên phù hợp hơn cho việc tăng cường dữ liệu.

Trong bài báo này, chúng tôi đề xuất một khung tăng cường dựa trên khuếch tán mới cho phân đoạn nhân tế bào. Phương pháp được đề xuất bao gồm hai bước: tổng hợp cấu trúc nhân tế bào không điều kiện và tổng hợp ảnh mô bệnh học có điều kiện. Chúng tôi phát triển một mô hình khuếch tán không điều kiện và một mô hình khuếch tán có điều kiện cấu trúc nhân tế bào (Hình 1) cho bước thứ nhất và thứ hai tương ứng. Trong giai đoạn huấn luyện, chúng tôi huấn luyện mô hình khuếch tán không điều kiện sử dụng các cấu trúc nhân tế bào được tính toán từ các bản đồ thực thể và mô hình khuếch tán có điều kiện sử dụng các cặp ảnh và cấu trúc nhân tế bào. Trong giai đoạn kiểm tra, các cấu trúc nhân tế bào và các ảnh tương ứng được tạo ra liên tiếp bởi hai mô hình. Theo hiểu biết của chúng tôi, chúng tôi là người đầu tiên áp dụng các mô hình khuếch tán vào tăng cường ảnh mô bệnh học cho phân đoạn nhân tế bào.

**Đóng góp của chúng tôi là:**
1. Một khung tăng cường dữ liệu dựa trên khuếch tán có thể tạo ra ảnh mô bệnh học và nhãn phân đoạn của chúng từ đầu
2. Một mô hình tổng hợp cấu trúc nhân tế bào không điều kiện và một mô hình tổng hợp ảnh mô bệnh học có điều kiện
3. Các thí nghiệm cho thấy rằng với phương pháp của chúng tôi, bằng cách tăng cường chỉ 10% dữ liệu huấn luyện được gán nhãn, người ta có thể thu được kết quả phân đoạn tương đương với baseline được giám sát hoàn toàn

---

## 2. Phương Pháp

Mục tiêu của chúng tôi là tăng cường một tập dữ liệu chứa một số lượng hạn chế các ảnh được gán nhãn với nhiều mẫu hơn để cải thiện hiệu suất phân đoạn. Để tăng sự đa dạng của các ảnh được gán nhãn, chúng ta ưu tiên tổng hợp cả ảnh và các bản đồ thực thể tương ứng của chúng. Chúng tôi đề xuất một chiến lược hai bước để tạo ra các ảnh được gán nhãn mới. Cả hai bước đều dựa trên các mô hình khuếch tán. Tổng quan về khung được đề xuất được thể hiện trong Hình 2. Trong phần này, chúng tôi giới thiệu chi tiết hai bước.

### 2.1 Tổng Hợp Cấu Trúc Nhân Tế Bào Không Điều Kiện

Trong bước đầu tiên, chúng tôi nhắm đến việc tổng hợp thêm các bản đồ thực thể. Vì không khả thi để trực tiếp tạo ra một bản đồ thực thể, thay vào đó chúng tôi chọn tạo ra sự thay thế của nó - cấu trúc nhân tế bào, được định nghĩa là sự kết hợp của ngữ nghĩa cấp pixel và biến đổi khoảng cách. Ngữ nghĩa cấp pixel là một bản đồ nhị phân trong đó 1 hoặc 0 chỉ ra liệu một pixel có thuộc về một nhân hay không. Biến đổi khoảng cách bao gồm biến đổi khoảng cách ngang và dọc, được tính bằng cách tính khoảng cách chuẩn hóa của mỗi pixel trong một nhân đến đường ngang và đường dọc đi qua tâm nhân [5]. Rõ ràng, cấu trúc nhân là một bản đồ 3 kênh có cùng kích thước với ảnh. Vì các thực thể nhân có thể được xác định từ cấu trúc nhân, chúng ta có thể dễ dàng xây dựng bản đồ thực thể tương ứng bằng cách thực hiện thuật toán watershed được điều khiển bởi marker trên cấu trúc nhân [29]. Do đó, vấn đề tổng hợp bản đồ thực thể chuyển sang tổng hợp cấu trúc nhân. Chúng tôi triển khai một mô hình khuếch tán không điều kiện để học phân phối của các cấu trúc nhân.

Ký hiệu một cấu trúc nhân thực là y₀, được lấy mẫu từ phân phối thực q(y). Để tối đa hóa hàm hợp lý dữ liệu, mô hình khuếch tán định nghĩa một quá trình thuận và một quá trình ngược. Trong quá trình thuận, một lượng nhỏ nhiễu Gauss được thêm liên tiếp vào mẫu y₀ trong T bước bằng:

```
yₜ = √(1-βₜ)yₜ₋₁ + √βₜεₜ₋₁, t = 1,...,T    (1)
```

trong đó εₜ ~ N(0,I) và {βₜ ∈ (0,1)}ᵀₜ₌₁ là một lịch trình phương sai. Chuỗi kết quả {y₀,...,yₜ} tạo thành một chuỗi Markov. Xác suất có điều kiện của yₜ cho trước yₜ₋₁ tuân theo phân phối Gauss:

```
q(yₜ|yₜ₋₁) = N(yₜ; √(1-βₜ)yₜ₋₁, βₜI)    (2)
```

Trong quá trình ngược, vì q(yₜ₋₁|yₜ) không thể dễ dàng ước tính, một mô hình p_θ(yₜ₋₁|yₜ) (thường là một mạng nơ-ron) sẽ được học để xấp xỉ q(yₜ₋₁|yₜ). Cụ thể, p_θ(yₜ₋₁|yₜ) cũng là một phân phối Gauss:

```
p_θ(yₜ₋₁|yₜ) = N(yₜ₋₁; μ_θ(yₜ,t), Σ_θ(yₜ,t))    (3)
```

Hàm mục tiêu là tổn thất giới hạn dưới biến phân: L = Lₜ + Lₜ₋₁ + ... + L₀, trong đó mọi số hạng ngoại trừ L₀ là phân kỳ KL giữa hai phân phối Gauss. Trong thực tế, một phiên bản đơn giản hóa của Lₜ thường được sử dụng [8]:

```
L^simple_t = E_{y₀,εₜ}‖εₜ - ε_θ(√(ᾱₜ)yₜ + √(1-ᾱₜ)εₜ, t)‖²    (4)
```

trong đó αₜ = 1 - βₜ và ᾱₜ = ∏ᵢ₌₁ᵗ αᵢ. Rõ ràng, mục tiêu tối ưu hóa của mạng nơ-ron được tham số hóa bởi θ là dự đoán nhiễu Gauss εₜ từ đầu vào yₜ tại thời điểm t.

Sau khi mạng được huấn luyện, người ta có thể giảm nhiễu dần dần một điểm ngẫu nhiên từ N(0,I) bằng T bước để tạo ra một mẫu mới:

```
yₜ₋₁ = (1/√αₜ)(yₜ - (1-αₜ)/√(1-ᾱₜ) ε_θ(yₜ,t)) + σₜz, z ~ N(0,I)    (5)
```

Để tổng hợp các cấu trúc nhân, chúng tôi huấn luyện một DDPM không điều kiện trên các cấu trúc nhân được tính toán từ các bản đồ thực thể thực. Theo [8], mạng của DDPM không điều kiện này có kiến trúc U-Net.

### 2.2 Tổng Hợp Ảnh Mô Bệnh Học Có Điều Kiện

Trong bước thứ hai, chúng tôi tổng hợp các ảnh mô bệnh học có điều kiện trên các cấu trúc nhân. Không có bất kỳ ràng buộc nào, một mô hình khuếch tán không điều kiện sẽ tạo ra các mẫu đa dạng. Thường có hai cách để tổng hợp ảnh bị ràng buộc bởi các điều kiện nhất định: khuếch tán dẫn đường bởi bộ phân loại [3] và hướng dẫn không có bộ phân loại [10]. Vì khuếch tán dẫn đường bởi bộ phân loại yêu cầu huấn luyện một bộ phân loại riêng biệt là một chi phí bổ sung, chúng tôi chọn hướng dẫn không có bộ phân loại để kiểm soát quá trình lấy mẫu.

Cho ε_θ(xₜ,t) và ε_θ(xₜ,t,y) là các bộ dự đoán nhiễu của mô hình khuếch tán không điều kiện p_θ(x|y) và mô hình khuếch tán có điều kiện p_θ(x), tương ứng. Hai mô hình có thể được học với một mạng nơ-ron. Cụ thể, p_θ(x|y) được huấn luyện trên dữ liệu cặp (x₀,y₀) và p_θ(x) có thể được huấn luyện bằng cách loại bỏ ngẫu nhiên y (tức là y = ∅) với một tỷ lệ drop nhất định ∈(0,1) sao cho mô hình học tạo sinh không điều kiện và có điều kiện đồng thời. Bộ dự đoán nhiễu ε'_θ(xₜ,t,y) của hướng dẫn không có bộ phân loại là một sự kết hợp của hai bộ dự đoán trên:

```
ε'_θ(xₜ,t,y) = (w + 1)ε_θ(xₜ,t,y) - wε_θ(xₜ,t)    (6)
```

trong đó ε_θ(xₜ,t) = ε_θ(xₜ,t,y = ∅), w là một vô hướng kiểm soát cường độ của hướng dẫn không có bộ phân loại.

Không giống như mạng của tổng hợp cấu trúc nhân không điều kiện có đầu vào là cấu trúc nhân có nhiễu yₜ và đầu ra là dự đoán của εₜ(yₜ,t), mạng của tổng hợp ảnh nhân có điều kiện lấy ảnh nhân có nhiễu xₜ và cấu trúc nhân tương ứng y làm đầu vào và dự đoán của εₜ(xₜ,t,y) làm đầu ra. Do đó, mạng có điều kiện nên được trang bị khả năng căn chỉnh tốt cặp ảnh mô bệnh học và cấu trúc nhân. Vì cấu trúc nhân và ảnh mô bệnh học có các không gian đặc trưng khác nhau, việc đơn giản nối hoặc chuyển chúng qua một mô-đun cross-attention [7, 15, 17] trước khi vào U-Net sẽ làm giảm độ trung thực của ảnh và tạo ra sự tương ứng không rõ ràng giữa ảnh nhân tổng hợp và cấu trúc nhân của nó. Lấy cảm hứng từ [28], chúng tôi nhúng thông tin của cấu trúc nhân vào các bản đồ đặc trưng của ảnh nhân bằng mô-đun chuẩn hóa thích ứng không gian (SPADE) [25]. Nói cách khác, thông tin không gian và hình thái của nhân điều chỉnh các bản đồ đặc trưng được chuẩn hóa sao cho các nhân được tạo ra ở đúng vị trí trong khi nền được tạo tự do. Chúng tôi bao gồm mô-đun SPADE ở các cấp độ khác nhau của mạng để sử dụng thông tin đa tỷ lệ của cấu trúc nhân. Mạng của tổng hợp ảnh nhân có điều kiện cũng áp dụng kiến trúc U-Net. Bộ mã hóa là một chồng các Resblock và các khối attention (AttnBlocks). Mỗi Resblock bao gồm 2 GroupNorm-SiLU-Conv và mỗi Attnblock tính toán self-attention của bản đồ đặc trưng đầu vào. Bộ giải mã là một chồng các CondResBlock và các khối attention. Mỗi CondResBlock bao gồm SPADE-SiLU-Conv nhận cả bản đồ đặc trưng và cấu trúc nhân làm đầu vào.

---
![alt text](<../images/hinh 2.png>)
**Hình 2:** Khung tăng cường dữ liệu dựa trên khuếch tán được đề xuất. Đầu tiên chúng tôi tạo ra một cấu trúc nhân với mô hình khuếch tán không điều kiện, sau đó tạo ra ảnh có điều kiện trên cấu trúc nhân. Bản đồ thực thể từ cấu trúc nhân được ghép cặp với ảnh tổng hợp để tạo thành một mẫu mới.

**Bước 1:** Tổng hợp cấu trúc nhân không điều kiện
**Bước 2:** Tổng hợp ảnh nhân có điều kiện

---

## 3. Thí Nghiệm và Kết Quả

### 3.1 Chi Tiết Triển Khai

**Tập dữ liệu.** Chúng tôi tiến hành thí nghiệm trên hai tập dữ liệu: MoNuSeg [13] và Kumar [14]. Tập dữ liệu MoNuSeg có 44 ảnh được gán nhãn với kích thước 1000×1000, 30 cho huấn luyện và 14 cho kiểm tra. Tập dữ liệu Kumar bao gồm 30 ảnh được gán nhãn 1000×1000 từ bảy cơ quan của cơ sở dữ liệu The Cancer Genome Atlas (TCGA). Tập dữ liệu được chia thành 16 ảnh huấn luyện và 14 ảnh kiểm tra.

**Tổng hợp mẫu cặp.** Để xác thực hiệu quả của phương pháp tăng cường được đề xuất, chúng tôi tạo 4 tập con của mỗi tập dữ liệu huấn luyện với 10%, 20%, 50% và 100% nhãn thực thể nhân. Cụ thể, trước tiên chúng tôi cắt tất cả các ảnh của mỗi tập dữ liệu thành các patch 256×256 với bước nhảy 128, sau đó lấy các đặc trưng của tất cả các patch với ResNet50 được huấn luyện trước [6] và phân cụm các patch thành 6 lớp bằng K-means. Các patch gần với tâm cụm được chọn. Bộ mã hóa và bộ giải mã của hai mạng có 6 lớp với các kênh 256, 256, 512, 512, 1024 và 1024. Đối với mạng tổng hợp cấu trúc nhân không điều kiện, mỗi lớp của bộ mã hóa và bộ giải mã có 2 ResBlock và 3 lớp cuối chứa AttnBlock. Mạng được huấn luyện sử dụng bộ tối ưu AdamW với tốc độ học 10⁻⁴ và kích thước batch là 4. Đối với mạng tổng hợp ảnh mô bệnh học có điều kiện, mỗi lớp của bộ mã hóa và bộ giải mã có 2 ResBlock và 2 CondResBlock tương ứng, và 3 lớp cuối chứa AttnBlock. Mạng đầu tiên được huấn luyện theo phong cách hoàn toàn có điều kiện (tỷ lệ drop = 0) và sau đó được tinh chỉnh theo phong cách không có bộ phân loại (tỷ lệ drop = 0.2). Chúng tôi sử dụng bộ tối ưu AdamW với tốc độ học 10⁻⁴ và 2×10⁻⁵ cho hai giai đoạn huấn luyện, tương ứng. Kích thước batch được đặt là 1. Đối với quá trình khuếch tán của cả hai bước, chúng tôi đặt tổng số bước thời gian khuếch tán T là 1000 với một lịch trình phương sai tuyến tính {β₁,...,βₜ} theo [8].

Đối với tập dữ liệu MoNuSeg, chúng tôi tạo ra 512/512/512/1024 mẫu tổng hợp cho các tập con được gán nhãn 10%/20%/50%/100%; đối với tập dữ liệu Kumar, 256/256/256/512 mẫu tổng hợp được tạo ra cho các tập con được gán nhãn 10%/20%/50%/100%. Các cấu trúc nhân tổng hợp được tạo ra bởi mạng tổng hợp cấu trúc nhân và các ảnh tương ứng được tạo ra bởi mạng tổng hợp ảnh mô bệnh học với tỷ lệ hướng dẫn không có bộ phân loại w = 2. Mỗi mạng tuân theo quá trình khuếch tán ngược với 1000 bước thời gian [8]. Sau đó chúng tôi thu được các tập con được tăng cường bằng cách thêm các ảnh cặp tổng hợp vào các tập con được gán nhãn tương ứng.

**Phân đoạn nhân.** Hiệu quả của phương pháp tăng cường được đề xuất có thể được đánh giá bằng cách so sánh hiệu suất phân đoạn của việc sử dụng bốn tập con được gán nhãn và sử dụng các tập con được tăng cường tương ứng để huấn luyện một mô hình phân đoạn. Chúng tôi chọn huấn luyện hai mô hình phân đoạn nhân - Hover-Net [5] và PFF-Net [18]. Để định lượng hiệu suất phân đoạn, chúng tôi sử dụng hai chỉ số: hệ số Dice và Chỉ số Jaccard Tổng hợp (AJI) [14].

### 3.2 Hiệu Quả của Phương Pháp Tăng Cường Dữ Liệu Được Đề Xuất

Hình 3 cho thấy các mẫu tổng hợp từ các mô hình được huấn luyện trên tập con với 10% ảnh được gán nhãn. Chúng tôi có các quan sát sau. Thứ nhất, các mẫu tổng hợp trông thực tế: các mẫu của cấu trúc nhân tổng hợp và kết cấu của ảnh tổng hợp gần với các mẫu thực. Thứ hai, do cơ chế có điều kiện của mạng tổng hợp ảnh và lấy mẫu hướng dẫn bộ phân loại, các ảnh tổng hợp được căn chỉnh tốt với các cấu trúc nhân tương ứng, đây là điều kiện tiên quyết để trở thành các mẫu huấn luyện phân đoạn bổ sung. Thứ ba, các cấu trúc nhân và ảnh tổng hợp cho thấy sự đa dạng lớn: các mẫu tổng hợp giống với các phong cách khác nhau của các mẫu thực nhưng với sự khác biệt rõ ràng.

---
![alt text](<../images/hinh 3.png>)
**Hình 3:** Trực quan hóa các mẫu tổng hợp. Hàng thứ nhất và thứ hai cho thấy các patch được chọn và các cấu trúc nhân tương ứng từ tập con được gán nhãn 10% của tập dữ liệu MoNuSeg. Hàng thứ ba và thứ tư cho thấy các ảnh tổng hợp được chọn và các nhân tương ứng với phong cách tương tự như mẫu thực trong cùng cột.

- Hàng 1: Ảnh thực
- Hàng 2: Cấu trúc nhân thực  
- Hàng 3: Ảnh tổng hợp
- Hàng 4: Cấu trúc nhân tổng hợp

---

Sau đó chúng tôi huấn luyện các mô hình phân đoạn trên bốn tập con được gán nhãn của tập dữ liệu MoNuSeg và Kumar và các tập con được tăng cường tương ứng với cả ảnh thực và ảnh tổng hợp được gán nhãn. Với một tỷ lệ gán nhãn cụ thể, chẳng hạn 10%, chúng tôi đặt tên tập con gốc là tập con được gán nhãn 10% và tập được tăng cường là tập con được tăng cường 10%. Đặc biệt, tập con được gán nhãn 100% là baseline được giám sát hoàn toàn. Bảng 1 cho thấy hiệu suất phân đoạn với Hover-Net. Đối với tập dữ liệu MoNuSeg, rõ ràng là các chỉ số phân đoạn giảm với ít ảnh được gán nhãn hơn. Ví dụ, chỉ với 10% ảnh được gán nhãn, Dice và AJI giảm 2.4% và 3.1%, tương ứng. Tuy nhiên, bằng cách tăng cường tập con được gán nhãn 10%, Dice và AJI vượt qua baseline được giám sát hoàn toàn lần lượt 0.9% và 1.3%. Đối với trường hợp 20% và 50%, hai chỉ số thu được bởi tập con được tăng cường ở cùng mức với việc sử dụng tất cả các ảnh được gán nhãn. Lưu ý rằng các chỉ số của tập con được tăng cường 10% cao hơn so với tập con được tăng cường 20%, điều này có thể là do sự không xác định của việc huấn luyện và lấy mẫu mô hình khuếch tán. Thú vị là việc tăng cường tập dữ liệu đầy đủ cũng giúp ích: Dice tăng 1.3% và AJI tăng 1.6% so với tập dữ liệu đầy đủ ban đầu. Do đó, phương pháp tăng cường được đề xuất cải thiện hiệu suất phân đoạn một cách nhất quán với các tỷ lệ gán nhãn khác nhau.

---

**Bảng 1:** Hiệu quả của phương pháp tăng cường dữ liệu được đề xuất với Hover-Net.

| Dữ liệu huấn luyện | MoNuSeg | | Kumar | |
|---|---|---|---|---|
| | Dice | AJI | Dice | AJI |
| 10% có nhãn | 0.7969 | 0.6344 | 0.8040 | 0.5939 |
| 10% được tăng cường | 0.8291 | 0.6785 | 0.8049 | 0.6161 |
| 20% có nhãn | 0.8118 | 0.6501 | 0.8078 | 0.6098 |
| 20% được tăng cường | 0.8219 | 0.6657 | 0.8192 | 0.6255 |
| 50% có nhãn | 0.8182 | 0.6603 | 0.8175 | 0.6201 |
| 50% được tăng cường | 0.8291 | 0.6764 | 0.8158 | 0.6307 |
| 100% có nhãn | 0.8206 | 0.6652 | 0.8150 | 0.6183 |
| 100% được tăng cường | 0.8336 | 0.6810 | 0.8210 | 0.6301 |

---

Đối với tập dữ liệu Kumar, bằng cách tăng cường tập con được gán nhãn 10%, AJI tăng lên mức tương đương với việc sử dụng 100% ảnh được gán nhãn; bằng cách tăng cường tập con được gán nhãn 20% và 50%, các AJI vượt qua baseline được giám sát hoàn toàn. Những kết quả này chứng minh hiệu quả của phương pháp tăng cường được đề xuất rằng chúng ta có thể đạt được hiệu suất phân đoạn ở cùng mức hoặc cao hơn so với baseline được giám sát hoàn toàn bằng cách tăng cường một tập dữ liệu với một lượng nhỏ ảnh được gán nhãn.

**Tính tổng quát của phương pháp tăng cường dữ liệu được đề xuất.** Hơn nữa, chúng tôi có các quan sát tương tự khi sử dụng PFF-Net làm mô hình phân đoạn. Bảng 2 cho thấy kết quả phân đoạn với PFF-Net. Đối với cả hai tập dữ liệu MoNuSeg và Kumar, tất cả bốn tỷ lệ gán nhãn đều cải thiện đáng kể với các mẫu tổng hợp. Điều này chỉ ra tính tổng quát của phương pháp tăng cường được đề xuất của chúng tôi.

---

**Bảng 2:** Tính tổng quát của phương pháp tăng cường dữ liệu được đề xuất với PFF-Net.

| Dữ liệu huấn luyện | MoNuSeg | | Kumar | |
|---|---|---|---|---|
| | Dice | AJI | Dice | AJI |
| 10% có nhãn | 0.7489 | 0.5290 | 0.7685 | 0.5965 |
| 10% được tăng cường | 0.7764 | 0.5618 | 0.8051 | 0.6458 |
| 20% có nhãn | 0.7691 | 0.5629 | 0.7786 | 0.6087 |
| 20% được tăng cường | 0.7891 | 0.5927 | 0.8019 | 0.6400 |
| 50% có nhãn | 0.7663 | 0.5661 | 0.7797 | 0.6175 |
| 50% được tăng cường | 0.7902 | 0.5998 | 0.8104 | 0.6524 |
| 100% có nhãn | 0.7809 | 0.5708 | 0.8032 | 0.6461 |
| 100% được tăng cường | 0.7872 | 0.5860 | 0.8125 | 0.6550 |

---

## 4. Kết Luận

Trong bài báo này, chúng tôi đề xuất một phương pháp tăng cường dữ liệu dựa trên khuếch tán mới cho phân đoạn nhân trong ảnh mô bệnh học. Mô hình tổng hợp cấu trúc nhân không điều kiện được đề xuất có thể tạo ra các cấu trúc nhân với hình dạng nhân và phân phối không gian thực tế. Mô hình tổng hợp ảnh mô bệnh học có điều kiện được đề xuất có thể tạo ra các ảnh giống với ảnh mô bệnh học thực và có tính đa dạng cao. Sự căn chỉnh tốt giữa các ảnh tổng hợp và các cấu trúc nhân tương ứng được đảm bảo bằng thiết kế đặc biệt của mô hình khuếch tán có điều kiện và hướng dẫn không có bộ phân loại. Bằng cách tăng cường các tập dữ liệu với một lượng nhỏ ảnh được gán nhãn, chúng tôi đã đạt được kết quả phân đoạn thậm chí tốt hơn so với baseline được giám sát hoàn toàn trên một số benchmark. Công trình của chúng tôi chỉ ra tiềm năng lớn của các mô hình khuếch tán trong tổng hợp mẫu cặp cho ảnh mô bệnh học.

---

## Tài Liệu Tham Khảo
1. Amit, T., Shaharbany, T., Nachmani, E., Wolf, L.: Segdiff: Image segmentation
with diffusion probabilistic models. arXiv preprint arXiv:2112.00390 (2021)
2. Arjovsky, M., Chintala, S., Bottou, L.: Wasserstein generative adversarial networks.
In: International Conference on Machine Learning. pp. 214–223. PMLR (2017)
3. Dhariwal, P., Nichol, A.: Diffusion models beat gans on image synthesis. Advances
in Neural Information Processing Systems 34, 8780–8794 (2021)
4. Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair,
S., Courville, A., Bengio, Y.: Generative adversarial nets. Advances in Neural
Information Processing Systems 27 (2014)
5. Graham, S., Vu, Q.D., Raza, S.E.A., Azam, A., Tsang, Y.W., Kwak, J.T., Rajpoot,
N.: Hover-net: Simultaneous segmentation and classification of nuclei in multi-tissue
histology images. Medical image analysis 58, 101563 (2019)
6. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In:
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.
pp. 770–778 (2016)
7. He, X., Yang, S., Li, G., Li, H., Chang, H., Yu, Y.: Non-local context encoder:
Robust biomedical image segmentation against adversarial attacks. In: Proceedings
of the AAAI Conference on Artificial Intelligence. vol. 33, pp. 8417–8424 (2019)
8. Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. Advances in
Neural Information Processing Systems 33, 6840–6851 (2020)
9. Ho, J., Saharia, C., Chan, W., Fleet, D.J., Norouzi, M., Salimans, T.: Cascaded
diffusion models for high fidelity image generation. The Journal of Machine Learning
Research 23(1), 2249–2281 (2022)
10. Ho, J., Salimans, T.: Classifier-free diffusion guidance. In: NeurIPS 2021 Workshop
on Deep Generative Models and Downstream Applications (2021)
11. Isola, P., Zhu, J.Y., Zhou, T., Efros, A.A.: Image-to-image translation with condi-
tional adversarial networks. In: Proceedings of the IEEE conference on Computer
Vision and Pattern Recognition. pp. 1125–1134 (2017)
12. Karras, T., Laine, S., Aila, T.: A style-based generator architecture for generative
adversarial networks. In: Proceedings of the IEEE/CVF conference on Computer
Vision and Pattern Recognition. pp. 4401–4410 (2019)
10 X. Yu et al.
13. Kumar, N., Verma, R., Anand, D., Zhou, Y., Onder, O.F., Tsougenis, E., Chen,
H., Heng, P.A., Li, J., Hu, Z., et al.: A multi-organ nucleus segmentation challenge.
IEEE Transactions on Medical Imaging 39(5), 1380–1391 (2019)
14. Kumar, N., Verma, R., Sharma, S., Bhargava, S., Vahadane, A., Sethi, A.: A dataset
and a technique for generalized nuclear segmentation for computational pathology.
IEEE Transactions on Medical Imaging 36(7), 1550–1560 (2017)
15. Li, H., Chen, G., Li, G., Yu, Y.: Motion guided attention for video salient object
detection. In: Proceedings of the IEEE/CVF International Conference on Computer
Vision. pp. 7274–7283 (2019)
16. Li, H., Li, G., Lin, L., Yu, H., Yu, Y.: Context-aware semantic inpainting. IEEE
Transactions on Cybernetics 49(12), 4398–4411 (2018)
17. Li, H., Li, G., Yang, B., Chen, G., Lin, L., Yu, Y.: Depthwise nonlocal module for
fast salient object detection using a single thread. IEEE Transactions on Cybernetics
51(12), 6188–6199 (2020)
18. Liu, D., Zhang, D., Song, Y., Huang, H., Cai, W.: Panoptic feature fusion net: a
novel instance segmentation paradigm for biomedical and biological images. IEEE
Transactions on Image Processing 30, 2045–2059 (2021)
19. Liu, D., Zhang, D., Song, Y., Zhang, F., O’Donnell, L., Huang, H., Chen, M., Cai,
W.: Unsupervised instance segmentation in microscopy images via panoptic domain
adaptation and task re-weighting. In: Proceedings of the IEEE/CVF conference on
Computer Vision and Pattern Recognition. pp. 4243–4252 (2020)
20. Lou, W., Li, H., Li, G., Han, X., Wan, X.: Which pixel to annotate: a label-efficient
nuclei segmentation framework. IEEE Transactions on Medical Imaging 42(4),
947–958 (2022)
21. Lou, W., Yu, X., Liu, C., Wan, X., Li, G., Liu, S., Li, H.: Multi-stream cell
segmentation with low-level cues for multi-modality images. In: Competitions in
Neural Information Processing Systems. pp. 1–10. PMLR (2023)
22. Mirza, M., Osindero, S.: Conditional generative adversarial nets. arXiv preprint
arXiv:1411.1784 (2014)
23. Naylor, P., La ́e, M., Reyal, F., Walter, T.: Segmentation of nuclei in histopathology
images by deep regression of the distance map. IEEE Transactions on Medical
Imaging 38(2), 448–459 (2018)
24. Nichol, A.Q., Dhariwal, P., Ramesh, A., Shyam, P., Mishkin, P., Mcgrew, B.,
Sutskever, I., Chen, M.: Glide: Towards photorealistic image generation and editing
with text-guided diffusion models. In: International Conference on Machine Learning.
pp. 16784–16804. PMLR (2022)
25. Park, T., Liu, M.Y., Wang, T.C., Zhu, J.Y.: Semantic image synthesis with spatially-
adaptive normalization. In: Proceedings of the IEEE/CVF conference on Computer
Vision and Pattern Recognition. pp. 2337–2346 (2019)
26. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution
image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition. pp. 10684–10695 (2022)
27. Shaham, T.R., Dekel, T., Michaeli, T.: Singan: Learning a generative model from a
single natural image. In: Proceedings of the IEEE/CVF International Conference
on Computer Vision. pp. 4570–4580 (2019)
28. Wang, W., Bao, J., Zhou, W., Chen, D., Chen, D., Yuan, L., Li, H.: Semantic image
synthesis via diffusion models. arXiv preprint arXiv:2207.00050 (2022)
29. Yang, X., Li, H., Zhou, X.: Nuclei segmentation using marker-controlled water-
shed, tracking using mean-shift, and kalman filter in time-lapse microscopy. IEEE
Transactions on Circuits and Systems I: Regular Papers 53(11), 2405–2414 (2006)
Diffusion-based Data Augmentation for Nuclei Image Segmentation 11
30. Yue, J., Li, H., Wei, P., Li, G., Lin, L.: Robust real-world image super-resolution
against adversarial attacks. In: Proceedings of the 29th ACM International Confer-
ence on Multimedia. pp. 5148–5157 (2021)
31. Zhu, J.Y., Park, T., Isola, P., Efros, A.A.: Unpaired image-to-image translation using
cycle-consistent adversarial networks. In: Proceedings of the IEEE International
Conference on Computer Vision. pp. 2223–2232 (2017)