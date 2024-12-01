'''
Created on March 24, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class NGCF(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, args):
        super(NGCF, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.device = args.device
        self.emb_size = args.embed_size
        self.batch_size = args.batch_size
        self.node_dropout = args.node_dropout[0]
        self.mess_dropout = args.mess_dropout
        self.batch_size = args.batch_size

        self.norm_adj = norm_adj

        self.layers = eval(args.layer_size)
        self.decay = eval(args.regs)[0]
        # 가중치 감쇠

        """
        *********************************************************
        Init the weight of user-item.
        """
        self.embedding_dict, self.weight_dict = self.init_weight()

        """
        *********************************************************
        Get sparse adj.
        """
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)
        # .to -> GPU, CPU에 적재 및 데이터 타입 변환 가능

    def init_weight(self):
        # xavier init
        initializer = nn.init.xavier_uniform_
        # 글로럿 초기화
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_user,
                                                 self.emb_size))),
                                                # 행: n_user, 열: emb_size(default: 64)
                                                # 행: fan_out, 열: fan_in
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_item,
                                                 self.emb_size)))
                                                # 행: n_item, 열: emb_size(default: 64)
                                                # 행: fan_out, 열: fan_in
        })

        weight_dict = nn.ParameterDict()
        layers = [self.emb_size] + self.layers # layers = [64, 64, 64, 64]
        for k in range(len(self.layers)):
            weight_dict.update({'W_gc_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_gc_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

            weight_dict.update({'W_bi_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_bi_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

        return embedding_dict, weight_dict

    def _convert_sp_mat_to_sp_tensor(self, X):
        # scipy sparse matrix -> pytorch sparse matrix
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)
        # batch size 만큼의 scaler값을 갖는 1차원 텐서

        maxi = nn.LogSigmoid()(pos_scores - neg_scores)
        # 0보다 커질수록 0에 가까워지고 0보다 작아질수록 y = -x
        # pos_scores가 크면 0에 가깝고 반대면 음수

        mf_loss = -1 * torch.mean(maxi)
        # neg_scores가 클수록 큰 값을 가짐 => pos_scores가 크게 되도록 학습됨

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / self.batch_size
        # batch_size에 따라 값이 달라지지 않도록 조정, 정규화 정도를 decay로 조절

        return mf_loss + emb_loss, mf_loss, emb_loss

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())
        # (n_user x n_item) 선호도 return

    def forward(self, users, pos_items, neg_items, drop_flag=True):

        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    self.node_dropout,
                                    self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj
                                                # ._nnz() -> None-Zero Number
        # A_hat -> dropout 적용한 희소 행렬
        # dropout은 신경망의 노드에 적용하는것 아닌가? 인접행렬에 적용하는 이유는?

        ego_embeddings = torch.cat([self.embedding_dict['user_emb'],
                                    self.embedding_dict['item_emb']], 0)
        # [user_emb[0], ... user_emb[n-1], item_emb[0], ... item_emb[n-1]]
        # item-user 초기 상태

        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            # 행렬 곱을 하여 희소 행렬로부터 embedding을 업데이트
            # 원본 인접 행렬 x embedding
            # 원본 인접 행렬의 user-item간 연결 정보를 사용해 임베팅 정보 집계
            # (user_n+item_n, user_n+item_n) x (user_n+item_n, 64) -> (user_n+item_n, 64) matrix

            # transformed sum messages of neighbors.
            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) \
                                             + self.weight_dict['b_gc_%d' % k]
            # ((원본 인접 행렬 x embedding) x sum of graph convolution weight) + bias

            # bi messages of neighbors.
            # element-wise product
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            # embedding x (원본 인접 행렬 x embedding)
            # embedding의 독립성을 유지하면서 이웃 정보도 포함
            
            # transformed bi messages of neighbors.
            bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) \
                                            + self.weight_dict['b_bi_%d' % k]
            # ((embedding x (원본 인접 행렬 x embedding)) x sum of bi-interaction weight)) + bias

            # non-linear activation.
            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)
            # 집계 결과에 activation 함수 적용

            # message dropout.
            ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)
            # mess_dropout default: [0.1, 0.1, 0.1]

            # normalize the distribution of embeddings.
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            # l2노름 정규화

            all_embeddings += [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, 1)
        u_g_embeddings = all_embeddings[:self.n_user, :]
        i_g_embeddings = all_embeddings[self.n_user:, :]

        """
        *********************************************************
        look up.
        """
        u_g_embeddings = u_g_embeddings[users, :]
        # u_g_embeddings = [원본, layer 1, layer 2, layer3]
        # user embedding 결과가 모두 들어있음
        pos_i_g_embeddings = i_g_embeddings[pos_items, :]
        neg_i_g_embeddings = i_g_embeddings[neg_items, :]

        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
