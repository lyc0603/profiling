"""Script to visualize the embedding"""

# Load your model or KV
from gensim.models import Word2Vec, KeyedVectors
import random
from environ.constant import PROCESSED_DATA_PATH
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

kv = KeyedVectors.load(f"{PROCESSED_DATA_PATH}/model/w2v.kv")

LABEL = {
    # CEX
    "970057788": "Binance 20",
    "741828001": "Binance 7",
    "1322342753": "OKX 73",
    "223089734": "Upbit 1",
    "105432110": "Upbit 40",
    "62927793": "Upbit 41",
    "452799244": "Bitfinex 2",
    # DeFi
    "512386385": "Arbitrum: Bridge",
    "59458375": "Unswap V2 ETH-USDC",
    "631514182": "Unswap V2 PEPE-ETH",
    "791914138": "Uniswap V3 WBTC-ETH",
    "309226321": "Aave: Ethereum WETH V3",
    "377403277": "Aave: Ethereum WBTC V3",
    # Token
    "749195173": "Wrapped Ether",
    "500612774": "AAVE",
    # Null
    "1": "Null 1",
    "3117283": "Null 2",
}


# token -> index mapping
vocab_size = len(kv.key_to_index)

# randomly select 1000 vector to use PCA
random_indices = random.sample(range(vocab_size), 1_000_000)

vectors = [kv[kv.index_to_key[index]] for index in random_indices]
labels = [kv[k] for k, _ in LABEL.items()]

pca = PCA(n_components=2)
embed_2d = pca.fit_transform(labels + vectors)

plt.figure(figsize=(14, 14))
plt.scatter(embed_2d[:, 0], embed_2d[:, 1])
# Annotate the label
for i, label in enumerate(LABEL.values()):
    plt.annotate(label, (embed_2d[i, 0], embed_2d[i, 1]), rotation=45)

plt.show()
