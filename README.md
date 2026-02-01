# F023  五种推荐算法vue+flask电影推荐可视化系统

> 完整项目收费，可联系QQ: 81040295 微信: mmdsj186011 注明从git来的，谢谢！
也可以关注我的B站： 麦麦大数据 https://space.bilibili.com/1583208775
> 
B站up账号:  **麦麦大数据**
关注B站，有好处！
编号:  F023
## 视频

[video(video-UjMaydAV-1758179721636)(type-bilibili)(url-https://player.bilibili.com/player.html?aid=515019612)(image-https://i-blog.csdnimg.cn/img_convert/a13446bb37596afeb0cad2e00b6d3dcd.png)(title-vue 5种算法之电影推荐可视化系统 [ Vue+Python])]

## 1 系统简介
系统简介：本系统是一个基于Vue.js前端框架和Flask后端框架的电影推荐系统，使用MySQL作为数据库。该系统旨在为用户提供个性化的电影推荐服务，集成了五种推荐算法：基于用户的协同过滤（UserCF）、基于物品的协同过滤（ItemCF）、奇异值分解（SVD）、混合协同过滤（混合CF）以及结合协同过滤与深度学习的混合CF+神经网络。系统不仅能够根据用户的历史行为和偏好推荐电影，还能够通过数据可视化功能帮助用户更直观地了解电影信息。通过花瓣图、柱状图、折线图等图表，用户可以分析电影的相关数据；通过漏斗图、仪表盘等可视化工具，可以深入了解评分分布和趋势。此外，系统还提供了电影介绍的词云分析功能，方便用户快速了解电影主题。用户可以通过注册和登录功能管理个人信息，并享受个性化的推荐服务。
## 2 功能设计
推荐系统模块：实现了UserCF、ItemCF、SVD、混合CF和混合CF+神经网络五种推荐算法，能够根据用户的历史评分、观看记录以及电影的特征提供多样化的推荐结果。
数据分析模块：通过多种图表形式（如花瓣图、柱状图、折线图、漏斗图、仪表盘）对电影数据进行可视化分析，帮助用户了解电影的热度、评分分布、类型分布等信息。
电影分析模块：提供电影介绍的词云分析功能，帮助用户快速了解电影的主题和关键词。
用户模块：支持用户注册、登录和个人信息管理，确保用户数据的安全性和个性化推荐的基础数据。
评分分析模块：通过可视化工具展示电影的评分分布和趋势，为用户提供更直观的评分参考。
系统管理模块：提供后台管理功能，方便管理员维护电影数据、用户数据和推荐模型。
通过这些功能模块的协同工作，系统能够为用户提供高效、准确、个性化的电影推荐服务，同时帮助用户深入了解电影的多方面信息。。
### 2.1系统架构图
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b4dcb3955a4049949e232f1c9b682187.png)
### 2.2 功能模块图
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8c6149725a144250b74ee51cbaac4a36.png)
### 2.3 推荐算法
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3e1e95385abd4596953de5b04738766b.png)
## 3 功能展示
### 3.1 登录 & 注册
登录注册做的是一个可以切换的登录注册界面，点击去登录后者去注册可以切换，背景是一个视频，循环播放。
登录需要验证用户名和密码是否正确，如果**不正确会有错误提示**。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f3f0b9b6218d45d8881aa0484600b28f.png)
注册需要**验证用户名是否存在**，如果错误会有提示。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/64a48ed65ee4456183e7bfbe40d8522c.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ec5fd52eea854f96939053e317428179.png)
### 3.2 主页
主页的布局采用了左侧是菜单，右侧是操作面板的布局方法，右侧的上方还有用户的头像和退出按钮，如果是新注册用户，没有头像，这边则不显示，需要在个人设置中上传了头像之后就会显示。
数据统计包含了系统内各种类型的电影还有各个国家电影的统计情况：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/df42471de596497a9537d656696d2aa8.png)
### 3.3 推荐算法
**usercf**:
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8047c253f20c4ef29f3bdf7fcc00fe76.png)
**itemcf**:
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/399d51ea912a44de87178430bfa3200b.png)
**svd**:
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/60a0ee7f7bb2411fbf15bf656e317b4c.png)
**混合CF**:
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/de0a67c2a1ae4c6e8e538e6ae314452f.png)
**神经网络**：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/074b1e07d9d348b9a21cf7370c06d500.png)
### 3.4 数据分析
电影分析：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e2fd598e20874d8b82ff753a683a26cc.png)
评分分析：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bf98b0bb5db34ceca3681e9d2428c494.png)
电影地图：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/eb0b4bbf4c174170beed8920108ccf52.png)
词云分析：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c65a396084384fed98a4a856f11ad0a3.png)

## 4程序代码
### 4.1 代码说明
代码介绍：基于Python的MLP（多层感知器）豆瓣电影推荐算法是一种利用深度学习技术实现的个性化推荐系统。该算法通过分析用户在豆瓣平台上的历史行为数据（如评分、观看记录等），构建一个MLP神经网络模型，以预测用户对未观看电影的潜在评分或兴趣偏好。模型首先对用户和电影特征进行嵌入表示，然后通过多个全连接层进行非线性变换，最终输出预测评分。该推荐方法能够有效捕捉用户与电影之间的复杂交互关系，提高推荐准确性和用户体验，适用于大规模数据场景，并可通过调整网络结构优化性能。
### 4.2 流程图
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ea6f5fde98794799901bcb2df8452b0e.png)

### 4.3 代码实例
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 定义MLP模型
class MLPRecommendation(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=50, hidden_layers=[100, 50]):
        super(MLPRecommendation, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        
        # 构建MLP层：输入是用户和电影嵌入的拼接（维度为2*embedding_dim）
        layers = []
        input_dim = 2 * embedding_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))  # 输出层预测评分
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, user_ids, movie_ids):
        user_embedded = self.user_embedding(user_ids)
        movie_embedded = self.movie_embedding(movie_ids)
        concatenated = torch.cat([user_embedded, movie_embedded], dim=-1)
        output = self.mlp(concatenated)
        return output.squeeze()

# 自定义数据集类
class MovieDataset(Dataset):
    def __init__(self, user_ids, movie_ids, ratings):
        self.user_ids = torch.tensor(user_ids, dtype=torch.long)
        self.movie_ids = torch.tensor(movie_ids, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float)
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.user_ids[idx], self.movie_ids[idx], self.ratings[idx]

# 训练函数
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for user_ids, movie_ids, ratings in dataloader:
            optimizer.zero_grad()
            predictions = model(user_ids, movie_ids)
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}')

# 示例使用
if __name__ == "__main__":
    # 假设数据（实际应从文件加载，如CSV）
    num_users = 1000
    num_movies = 2000
    user_ids = np.random.randint(0, num_users, size=5000)  # 示例数据
    movie_ids = np.random.randint(0, num_movies, size=5000)
    ratings = np.random.uniform(1, 5, size=5000)
    
    dataset = MovieDataset(user_ids, movie_ids, ratings)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = MLPRecommendation(num_users, num_movies)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_model(model, dataloader, criterion, optimizer, num_epochs=5)
    
    # 生成推荐：预测用户对电影的评分并排序
    def recommend_movies(model, user_id, top_k=5):
        all_movies = torch.arange(num_movies, dtype=torch.long)
        user_tensor = torch.tensor([user_id] * num_movies, dtype=torch.long)
        with torch.no_grad():
            predictions = model(user_tensor, all_movies)
        top_indices = torch.topk(predictions, top_k).indices
        return top_indices.tolist()
    
    user_id_example = 0
    recommended = recommend_movies(model, user_id_example)
    print(f"Recommended movies for user {user_id_example}: {recommended}")

```
