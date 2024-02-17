import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

class MultiHeadAttention(nn.Module):
    #dmodel - dimension of input, headsNum - number of heads to share input
    def __init__(self, dModel, headsNum):
        super(MultiHeadAttention, self).__init__()
        assert dModel % headsNum == 0, "dimension must of a multiple of the number of heads"
        #model dimensions
        self.dModel = dModel
        self.headsNum = headsNum
        self.dK = dModel // headsNum

        #linear layers for query, key, value and output
        self.wQ = nn.Linear(dModel, dModel)
        self.wK = nn.Linear(dModel, dModel)
        self.wV = nn.Linear(dModel, dModel)
        self.wO = nn.Linear(dModel, dModel)

    def scaledDotProductAttention(self, query, key, value, mask=None):
        #score = q * k^T / sqrt(dK) (later normalize & softmax this)
        attentionScores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.dK)
        if mask is not None: #fill masked areas with negative infinity so it gets ignored
            attentionScores.masked_fill(mask == 0, -1e9)
        attentionProbabilites = torch.softmax(attentionScores, dim=-1) #FIXME why -1?
        product = torch.matmul(attentionProbabilites)
        return product
    
    def splitHeads(self, x):
        batchSize, seqLength, dModel = x.size()
        return x.view(batchSize, seqLength, self.headsNum, self.dK).transpose(1, 2)
    
    def combineHeads(self, x):
        batchSize, _, seqLength, dK = x.size()
        return x.tranpose(1, 2).contiguous().view(batchSize, seqLength, self.dModel)
    
    def forward(self, query, key, value, mask=None): #perform linear transformations
        query = self.splitHeads(self.wQ(query))
        key = self.splitHeads(self.wK(key))
        value = self.splitHeads(self.wV(value))

        attention = self.scaledDotProductAttention(query, value, key, mask)
        output = self.wO(self.combineHeads(attention))
        return output
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, dModel, dFF): #FF - fully connected
        super(PositionWiseFeedForward, self).__init__()
        #two fully connected layers
        self.fc1 = nn.Linear(dModel, dFF)
        self.fc2 = nn.Linear(dModel, dFF)
        self.reLU = nn.ReLU()

    def forward(self, x): #propagate through layers of network
        return self.fc2(self.relu(self.fc1(x)))
    
class PositionalEncoding(nn.Module):
    def __init__(self, dModel, maxSeqLength):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(maxSeqLength, dModel)
        position = torch.arange(0, maxSeqLength, dtype=torch.float).unsqueeze(1)
        divTerm = torch.exp(torch.arange(0, dModel, 2).float() * -(math.log(10000.0) / dModel))

        pe[:, 0::2] = torch.sin(position * divTerm)
        pe[:, 1::2] = torch.cos(position * divTerm)
        self.register_buffer('pe', pe.unsqueeze(0)) #use pe not as a trainable parameter
    
    def forward(self, x):
        return x + self.pe[:, x.size(1)]
    
class EncoderLayer(nn.Module):
    def __init__(self, dModel, headsNum, dFF, droput):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(dModel, headsNum)
        self.feedForward = PositionWiseFeedForward(dModel, dFF)
        self.norm1 = nn.LayerNorm(dModel)
        self.norm2 = nn.LayerNorm(dModel)
        self.dropout = nn.Dropout(droput)

    def forward(self, x, mask):
        attentionOutput = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attentionOutput))
        ffOutput = self.feedForward(x)
        x = self.norm2(x + self.dropout(ffOutput))
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, dModel, headsNum, dFF, dropout):
        super(DecoderLayer, self).__init__()
        self.selfAttention = MultiHeadAttention(dModel, headsNum)
        self.crossAttention = MultiHeadAttention(dModel, headsNum)
        self.feedForward = PositionWiseFeedForward(dModel, dFF)
        self.norm1 = nn.LayerNorm(dModel)
        self.norm2 = nn.LayerNorm(dModel)
        self.norm3 = nn.LayerNorm(dModel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encodedOutput, sourceMask, targetMask):
        attentionOutput = self.selfAttention(x, x, x, targetMask)
        x = self.norm1(x + self.dropout(attentionOutput))
        attentionOutput = self.crossAttention(x, encodedOutput, encodedOutput, sourceMask)
        x = self.norm2(x + self.dropout(attentionOutput))
        ffOutput = self.feedForward(x)
        x = self.norm3(x + self.dropout(ffOutput))
        return x
    
class Transformer(nn.Module):
    def __init__(self, sourceVocabSize, targetVocabSize, dModel, headsNum, numLayers,dFF, maxSeqLength, dropout):
        self.encoderEmbed = nn.Embedding(sourceVocabSize, dModel)
        self.decoderEmbed = nn.Embedding(targetVocabSize, dModel)
        self.positionEncoding = PositionalEncoding(dModel, maxSeqLength)

        self.encoderLayers = nn.ModuleList([EncoderLayer(dModel, headsNum, dFF, dropout) for _ in range(numLayers)])
        self.decoderLayers = nn.ModuleList([DecoderLayer(dModel, headsNum, dFF, dropout) for _ in range(numLayers)])

        self.fc = nn.Linear(dModel, targetVocabSize)
        self.dropout = nn.Dropout(dropout)

    def generateMask(self, source, target)
        sourceMask = (source != 0).unsqueeze(1).unsqueeze(2)
        targetMask = (target != 0).unsqueeze(1).unsqueeze(3)
