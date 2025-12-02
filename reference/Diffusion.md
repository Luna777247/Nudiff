# MÔ HÌNH XÁC SUẤT KHỬ NHIỄU LAN TỎA (Denoising Diffusion Probabilistic Models)

**Tác giả**: Jonathan Ho, Ajay Jain, Pieter Abbeel  
**Đơn vị**: UC Berkeley  
**arXiv**: 2006.11239v2 [cs.LG] – 16 Dec 2020  
**Hội nghị**: NeurIPS 2020, Vancouver, Canada

### Tóm tắt (Abstract)

Chúng tôi trình bày kết quả tổng hợp hình ảnh chất lượng cao bằng việc sử dụng **mô hình xác suất diffusion** – một lớp mô hình biến ẩn được lấy cảm hứng từ nhiệt động lực học không cân bằng.  

Kết quả tốt nhất đạt được bằng cách huấn luyện trên một **giới hạn biến phân có trọng số** được thiết kế dựa trên một kết nối mới giữa mô hình xác suất diffusion và **denoising score matching kết hợp với động lực học Langevin**.  

Mô hình của chúng tôi tự nhiên hỗ trợ một **lược đồ giải nén mất mát tiến triển** – có thể được hiểu như một **tổng quát hóa mạnh mẽ của giải mã tự hồi quy**.  

Trên CIFAR10 không điều kiện, chúng tôi đạt **Inception score 9.46** và **FID 3.17** (tốt nhất tại thời điểm công bố).  
Trên LSUN 256×256, chất lượng mẫu tương đương ProgressiveGAN.  

Mã nguồn: https://github.com/hojonathanho/diffusion

### 1. Giới thiệu

Các mô hình sinh gần đây (GANs, autoregressive, flow, VAE…) đã tạo ra hình ảnh và âm thanh rất ấn tượng. Tuy nhiên, cho đến nay chưa ai chứng minh được rằng **mô hình diffusion** (Sohl-Dickstein et al., 2015) có thể tạo ra mẫu chất lượng cao.

Bài báo này chứng minh điều đó:  
- Đạt chất lượng mẫu đôi khi **tốt hơn** cả các kết quả đã công bố của GANs.  
- Thiết lập một **liên hệ rõ ràng** giữa mô hình diffusion và **denoising score matching + Langevin dynamics** (Song & Ermon, 2019, 2020).  
- Parameterization mới này cho kết quả tốt nhất về chất lượng mẫu.  
- Mô hình tự nhiên thực hiện **nén mất mát tiến triển**, giải thích tại sao likelihood không cao nhưng mẫu vẫn đẹp.

### 2. Nền tảng

Mô hình diffusion là mô hình biến ẩn dạng:

pθ(x0) = ∫ pθ(x0:T) dx1:T

Quá trình ngược (reverse process) là một chuỗi Markov học được, bắt đầu từ prior chuẩn:

pθ(x₀:T) = p(xT) ∏_{t=1}^T pθ(x_{t−1}|x_t)  
pθ(x_{t−1}|x_t) = N(x_{t−1}; μθ(x_t, t), Σθ(x_t, t))

Quá trình thuận (forward process – diffusion process) là cố định, thêm dần nhiễu Gaussian theo lịch β₁…β_T:

q(x_{1:T}|x₀) = ∏_{t=1}^T q(x_t|x_{t−1})  
q(x_t|x_{t−1}) = N(x_t; √(1−β_t) x_{t−1}, β_t I)

Có thể lấy mẫu x_t trực tiếp từ x₀ ở bước t bất kỳ:

q(x_t|x₀) = N(x_t; √¯α_t x₀, (1−¯α_t)I)  
với α_t = 1−β_t, ¯α_t = ∏_{s=1}^t α_s

Huấn luyện bằng ELBO chuẩn:

L = E[-log pθ(x₀)] ≤ L_vb

### 3. Diffusion và Denoising Autoencoders

#### 3.1 Quá trình thuận (LT)

Chúng tôi cố định β_t (không học), nên LT là hằng số → bỏ qua.

#### 3.2 Quá trình ngược (L₁ đến L_{T−1})

Đặt Σθ(x_t, t) = σ²_t I (cố định, thử cả β_t và ˜β_t đều ổn).

Thay vì dự đoán trực tiếp μθ, chúng tôi đề xuất dự đoán **nhiễu ϵ**:

x_t = √¯α_t x₀ + √(1−¯α_t) ϵ

→ μθ(x_t, t) = (1/√α_t) [x_t − (1−α_t)/√(1−¯α_t) ϵθ(x_t, t)]

Điều này biến việc huấn luyện thành:

L_{t−1} ≈ ||ϵ − ϵθ(√¯α_t x₀ + √(1−¯α_t)ϵ, t)||² × trọng số

→ **giống hệt denoising score matching đa tỷ lệ nhiễu**.

Và sampling giống **annealed Langevin dynamics** khi thêm nhiễu ở mỗi bước.

#### 3.3 Decoder rời rạc (L₀)

Dữ liệu ảnh được scale về [-1,1]. Ở bước cuối, dùng decoder rời rạc (tích phân Gaussian trên bin).

#### 3.4 Mục tiêu huấn luyện đơn giản hóa (được dùng thực tế)

L_simple = E_{t,x₀,ϵ} [ ||ϵ − ϵθ(√¯α_t x₀ + √(1−¯α_t)ϵ, t)||² ]

Đây là **mục tiêu đơn giản nhất và cho kết quả mẫu tốt nhất**.

### 4. Thí nghiệm

T = 1000, β_t tăng tuyến tính từ 10⁻⁴ → 0.02.  
Kiến trúc: U-Net với group norm + self-attention ở 16×16.

#### 4.1 Chất lượng mẫu

CIFAR10 không điều kiện:  
- FID = **3.17**  
- IS = **9.46**

Tốt hơn hầu hết mô hình khác tại thời điểm 2020 (chỉ thua StyleGAN2-ADA sau này).

LSUN 256×256: FID ~4–8, ngang ProgressiveGAN.

#### 4.2 Ablation

- Dự đoán ϵ + L_simple → tốt nhất.  
- Dự đoán μ → kém hơn.  
- Học Σθ → không ổn định.

#### 4.3 Nén tiến triển (Progressive coding)

Mô hình rất tốt ở nén mất mát

Rate ≈ 1.78 bit/dim, distortion ≈ 1.97 bit/dim → RMSE ~0.95/255 (gần như không thấy bằng mắt).

Nếu decode tiến triển từ x_T → x₀, chất lượng tăng dần rất đẹp (hình 6).

### Kết luận chính của bài báo (tóm gọn)

1. Diffusion model + ϵ-parameterization + L_simple → chất lượng mẫu state-of-the-art 2020.  
2. Liên hệ rõ ràng với denoising score matching và Langevin dynamics.  
3. Mô hình tự nhiên thực hiện **nén mất mát tiến triển** (progressive lossy compression), giải thích tại sao likelihood không cạnh tranh nhưng mẫu cực đẹp.

Đây là bài báo đặt nền móng cho toàn bộ làn sóng Diffusion hiện nay (Stable Diffusion, DALL·E 3, Sora, v.v.). Năm 2020 rất ít người tin diffusion sẽ thắng GAN, nhưng DDPM đã chứng minh điều đó.