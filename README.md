{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "yhkE4uU0IEkQ",
    "outputId": "cb0ca221-05f9-4bc7-ad11-71f923227294"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Drive already mounted at gdrive; to attempt to forcibly remount, call drive.mount(\"gdrive\", force_remount=True).\n"
     ]
    }
   ],
   "source": [
    "from google.colab import drive\n",
    "drive.mount(\"gdrive\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "KNt_KPyGG6sh",
    "outputId": "2fe6a6df-2648-49ba-cac0-74bdd76e9400"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package punkt to /root/nltk_data...\n",
      "[nltk_data]   Package punkt is already up-to-date!\n",
      "[nltk_data] Downloading package stopwords to /root/nltk_data...\n",
      "[nltk_data]   Package stopwords is already up-to-date!\n"
     ]
    }
   ],
   "source": [
    "import re\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import string\n",
    "import nltk\n",
    "nltk.download('punkt')\n",
    "from nltk.tokenize import word_tokenize, sent_tokenize\n",
    "nltk.download('stopwords')\n",
    "from nltk.corpus import stopwords\n",
    "stopwords = set(stopwords.words('turkish'))\n",
    "from typing import List\n",
    "\n",
    "import torch.nn"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "s4zxTB-dIVP6"
   },
   "source": [
    "I am going to use a turkish news dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "id": "Fl6F90erIU3X"
   },
   "outputs": [],
   "source": [
    "train_df = pd.read_csv(\"/content/gdrive/MyDrive/CS-445/train.csv\",index_col=[0])\n",
    "test_df = pd.read_csv(\"/content/gdrive/MyDrive/CS-445/test.csv\",index_col=[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "id": "aaxQTImOIoNm"
   },
   "outputs": [],
   "source": [
    "result = pd.concat([train_df,test_df])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "id": "o5t-A8dwFmuO"
   },
   "outputs": [],
   "source": [
    "WINDOW_SIZE = 3  # 3 words to the left, 3 to the right\n",
    "CORPUS = result[\"text\"].to_list()[:1000]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "id": "yTnse2k7HXAn"
   },
   "outputs": [],
   "source": [
    "def prune(text=\"\",punc=string.punctuation.replace(\".\",\"\"),stopwords=stopwords,lower=True)->List[str]:\n",
    "  ''' \n",
    "    This function initially prunes a text from punctuation and stopwords and finally tokenizes a text.\n",
    "    Parameters:\n",
    "      text: (str) This is the text that will be processed.\n",
    "      \n",
    "      punc: (str) This is a string of punctuation characters. \n",
    "      Default is string.punctuation.\n",
    "      \n",
    "      stopwords: (List[str]) This is a list of turkish stop word strings.\n",
    "      Default is nltk.stopwords.\n",
    "\n",
    "    Returns:\n",
    "      tmp: (List[str]) This is a list of tokenized and pruned text as string.\n",
    "  '''\n",
    "  for p in punc:\n",
    "    text = text.replace(p,\" \")\n",
    "  text = text.lower() if(lower) else text\n",
    "  tokenized = word_tokenize(text)\n",
    "  tmp = []\n",
    "  for token in tokenized:\n",
    "    if(token not in stopwords):\n",
    "      tmp.append(token)\n",
    "  return tmp"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "LydrkWNaGBq2"
   },
   "source": [
    "First we should process corpus such that resulting output should be a list of tokens with order."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "fPi0J3z3J0ak",
    "outputId": "699b2616-e798-49ee-df61-fc6ed81fc88e"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['beşiktaş',\n",
       " 'ın',\n",
       " 'eski',\n",
       " 'teknik',\n",
       " 'direktörü',\n",
       " 'slaven',\n",
       " 'bilic',\n",
       " 'türkiye',\n",
       " 'hırvatistan',\n",
       " 'maçında',\n",
       " 'yorumculuk',\n",
       " 'yapmak',\n",
       " 'üzere',\n",
       " 'lig',\n",
       " 'tv',\n",
       " 'anlaştı',\n",
       " '.',\n",
       " 'euro',\n",
       " '2016',\n",
       " 'nın',\n",
       " 'yayıncı',\n",
       " 'kuruluşlarından',\n",
       " 'olan',\n",
       " 'lig',\n",
       " 'tv',\n",
       " 'türkiye',\n",
       " 'nin',\n",
       " 'd',\n",
       " 'grubu',\n",
       " 'nda',\n",
       " 'hırvatistan',\n",
       " 'oynayacağı',\n",
       " 'ilk',\n",
       " 'maç',\n",
       " 'slaven',\n",
       " 'bilic',\n",
       " 'anlaşıldığını',\n",
       " 'duyurdu',\n",
       " '.',\n",
       " 'beşiktaş',\n",
       " 'ın',\n",
       " 'eski',\n",
       " 'teknik',\n",
       " 'direktörü',\n",
       " 'slaven',\n",
       " 'bilic',\n",
       " '12',\n",
       " 'haziran',\n",
       " 'pazar',\n",
       " 'günü',\n",
       " 'tsi̇',\n",
       " '16',\n",
       " '00',\n",
       " 'başlayacak',\n",
       " 'mücadelede',\n",
       " 'yorumcu',\n",
       " 'olacak',\n",
       " '.',\n",
       " 'lig',\n",
       " 'tv',\n",
       " 'slaven',\n",
       " 'bilic',\n",
       " 'in',\n",
       " 'yanı',\n",
       " 'sıra',\n",
       " 'a',\n",
       " 'milli',\n",
       " 'takım',\n",
       " 'ın',\n",
       " 'efsane',\n",
       " 'kalecilerinden',\n",
       " 'rüştü',\n",
       " 'reçber',\n",
       " 'in',\n",
       " 'karşılaşmanın',\n",
       " 'yorumcularından',\n",
       " 'olacağını',\n",
       " 'açıkladı',\n",
       " '.',\n",
       " 'euro',\n",
       " '2008',\n",
       " 'hırvatistan',\n",
       " 'ın',\n",
       " 'teknik',\n",
       " 'direktörü',\n",
       " 'olan',\n",
       " 'slaven',\n",
       " 'bilic',\n",
       " 'çeyrek',\n",
       " 'finalde',\n",
       " 'türkiye',\n",
       " 'ye',\n",
       " 'rakip',\n",
       " 'olmuş',\n",
       " '120',\n",
       " 'dakikası',\n",
       " '1',\n",
       " '1',\n",
       " 'biten',\n",
       " 'maçta',\n",
       " 'a',\n",
       " 'milli',\n",
       " 'takımımıza',\n",
       " 'penaltılarda',\n",
       " 'elenmişti',\n",
       " '.']"
      ]
     },
     "execution_count": 10,
     "metadata": {
      "tags": []
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "prune(text=CORPUS[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "id": "tArUqieKGmwb"
   },
   "outputs": [],
   "source": [
    "PROCESSED_CORPUS = []\n",
    "for text in CORPUS:\n",
    "  PROCESSED_CORPUS.extend(prune(text=text))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "id": "RIGbxmBeKsha"
   },
   "outputs": [],
   "source": [
    "vocabulary = set(PROCESSED_CORPUS)\n",
    "vocabulary_size = len(vocabulary)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "po2zU0n8KyFd",
    "outputId": "ac7fe787-055f-4314-b1c6-0fe9c2c31739"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['payet',\n",
       " 'canlandırmanın',\n",
       " 'liselerinde',\n",
       " 'gecikmeler',\n",
       " 'heykel',\n",
       " 'çalışacağını',\n",
       " 'gençliğini',\n",
       " 'i̇nsanlığın',\n",
       " 'kamuoyuna',\n",
       " 'trilyon',\n",
       " 'hakkınız',\n",
       " 'basıyordum',\n",
       " 'kayyum',\n",
       " 'elbiseli',\n",
       " 'hayvanların',\n",
       " 'zirvesinde',\n",
       " 'darba',\n",
       " 'geçtikten',\n",
       " 'yetmedi',\n",
       " 'anayla']"
      ]
     },
     "execution_count": 13,
     "metadata": {
      "tags": []
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "list(vocabulary)[:20]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "gsmRo6ooGAv6",
    "outputId": "ae0f6163-e680-4151-e722-94131083870a"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Context: ['beşiktaş', 'ın', 'eski', 'direktörü', 'slaven', 'bilic']\n",
      "Target: teknik\n"
     ]
    }
   ],
   "source": [
    "word2index = {word: i for i, word in enumerate(vocabulary)}\n",
    "data = [] # Will contain the focus and context words\n",
    "for i in range(3, len(PROCESSED_CORPUS) - 3):\n",
    "    context = [\n",
    "               PROCESSED_CORPUS[i - 3], \n",
    "               PROCESSED_CORPUS[i - 2], \n",
    "               PROCESSED_CORPUS[i - 1],\n",
    "               PROCESSED_CORPUS[i + 1], \n",
    "               PROCESSED_CORPUS[i + 2],\n",
    "               PROCESSED_CORPUS[i + 3], \n",
    "               ]\n",
    "\n",
    "    target = PROCESSED_CORPUS[i]\n",
    "    data.append((context, target))\n",
    "\n",
    "print(\"Context:\" , data[0][0])\n",
    "print(\"Target:\" , data[0][1])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Jw8kiOIkLgE3"
   },
   "source": [
    "We are trying to obtain a special format where there exists a focues no"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "moQGEL9SM-ev",
    "outputId": "ec06d0c9-861e-419a-e73f-aa42d8d30dd9"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "tensor([ 7446,  1893, 30992, 27658,  7406, 35977])"
      ]
     },
     "execution_count": 15,
     "metadata": {
      "tags": []
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "def make_context_vector(context, word2index):\n",
    "    idxs = [word2index[w] for w in context]\n",
    "    return torch.tensor(idxs, dtype=torch.long)\n",
    "\n",
    "make_context_vector(data[0][0],word2index)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "H0sFjr4zM02b",
    "outputId": "7e44d099-83ef-43c8-cc01-10fa74fc83fc"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['beşiktaş', 'ın', 'eski', 'direktörü', 'slaven', 'bilic']"
      ]
     },
     "execution_count": 16,
     "metadata": {
      "tags": []
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data[0][0]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "vDxDELy8M_y1"
   },
   "source": [
    "This might look complex but actually we are just encoding the context list with respect to its id and then repond it as a tensor."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "id": "-oUduh_2LaUe"
   },
   "outputs": [],
   "source": [
    "class CBOW(torch.nn.Module):\n",
    "    def __init__(self,vocab_size,embedding_dim):\n",
    "        super(CBOW, self).__init__()\n",
    "        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)\n",
    "        self.linear = torch.nn.Linear(embedding_dim, vocabulary_size)\n",
    "        self.log_softmax = torch.nn.LogSoftmax(dim = -1)\n",
    "\n",
    "    def forward(self, inputs):\n",
    "        embeds = sum(self.embeddings(inputs)).view(1,-1)\n",
    "        out = self.linear(embeds) \n",
    "        log_probs = self.log_softmax(out) \n",
    "\n",
    "        return log_probs\n",
    "\n",
    "    def word_emdedding(self, word):\n",
    "        word = torch.tensor([word2index[word]])\n",
    "        return self.embeddings(word).view(1,-1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "id": "myJZpOnhQ1wW"
   },
   "outputs": [],
   "source": [
    "model = CBOW(vocabulary_size, 2)\n",
    "\n",
    "loss_function = torch.nn.NLLLoss()\n",
    "optimizer = torch.optim.SGD(model.parameters(), lr=0.001)\n",
    "\n",
    "for epoch in range(100):\n",
    "    total_loss = 0\n",
    "\n",
    "    for context, target in data[:100]:\n",
    "        context_vector = make_context_vector(context, word2index)  \n",
    "        log_probs = model(context_vector)\n",
    "        tensor = torch.tensor([word2index[target]])#.to('cuda:0')\n",
    "        total_loss += loss_function(log_probs, tensor)\n",
    "    optimizer.zero_grad()\n",
    "    total_loss.backward()\n",
    "    optimizer.step()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "VLJIwshwX9A8",
    "outputId": "db4b8f78-2f3f-40b8-97ee-1335ab8ce19c"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(['beşiktaş', 'ın', 'eski', 'direktörü', 'slaven', 'bilic'], 'teknik')"
      ]
     },
     "execution_count": 19,
     "metadata": {
      "tags": []
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "id": "5dlahF65bh84"
   },
   "outputs": [],
   "source": [
    "index2word = {v: k for k, v in word2index.items()}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "SHZOEjxmVEEC",
    "outputId": "7a1db0ad-f7c9-4b5e-8586-2d416876d14c"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Context: ['beşiktaş']\n",
      "\n",
      "Prediction: slaven\n"
     ]
    }
   ],
   "source": [
    "context = ['beşiktaş']\n",
    "context_vector = make_context_vector(context, word2index)\n",
    "a = model(context_vector)\n",
    "\n",
    "#Print result\n",
    "print(f'Context: {context}\\n')\n",
    "print(f'Prediction: {index2word[torch.argmax(a[0]).item()]}')"
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "name": "pytorch-word2vec.ipynb",
   "provenance": [],
   "toc_visible": true
  },
  "kernelspec": {
   "display_name": "Python 3",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
