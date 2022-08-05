import json
import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()

stopwords_indo = open("../src/stopwords_indo.txt", "r").read().split(", ")
emoji_converter_indo = {'😛': 'menjulurkan lidah', '😠': 'marah', '💣': 'bom', '💔': 'patah hati', '😕': 'bingung', '😞': 'kecewa', '😑': 'tanpa ekspresi', '😋': 'enak', '😱': 'ketakutan', '😓': 'keringat dingin', '😮': 'tercengang', '😤': 'mengamuk', '😝': 'meledek', '😶': 'wajah tanpa mulut', '🔥': 'api', '☹': 'cemberut', '😬': 'meringis', '⚡': 'tegangan tinggi', '🤥': 'bohong', '😣': 'gigih', '🙇': 'membungkuk', '🏃': 'lari', '🐽': 'hidung babi', '😡': 'marah', '🙈': 'tutup mata', '🙁': 'agak cemberut', '🙊': 'tutup mulut', '🤔': 'berpikir', '👎': 'buruk', '👅': 'lidah', '😩': 'lelah', '🤐': 'tutup mulut', '😐': 'netral', '🙄': 'berpikir', '😏': 'seringai', '😥': 'kecewa namun lega', '😯': 'meredam', '😪': 'mengantuk', '😫': 'lelah', '😴': 'tidur', '😌': 'lega', '😜': 'menjulurkan lidah sambil mengedipkan mata', '🤤': 'mengidam', '😒': 'tidak terhibur', '😔': 'termenung', '🙃': 'wajah terbalik', '🤑': 'mata duitan', '😲': 'heran', '😖': 'terkutuk', '😟': 'khawatir', '😢': 'menangis', '😭': 'menangis tersedu', '😦': 'mengerutkan kening sambil buka mulut', '😧': 'menderita', '😨': 'takut', '😰': 'panik', '😳': 'wajah memerah', '😵': 'pusing', '😷': 'sakit pakai masker', '🤒': 'sakit pakai termometer', '🤕': 'sakit kepala perban', '🤢': 'tahan muntah', '🤧': 'bersin', '🤓': 'kutu buku', '😈': 'senyum jahat', '👿': 'iblis marah', '👹': 'ogre', '👺': 'goblin', '💀': 'tengkorak', '☠': 'tengkorak', '👻': 'hantu', '💩': 'kotoran', '🙀': 'kucing kaget', '😿': 'kucing menangis', '😾': 'kucing marah', '🙉': 'tutup telinga', '🙎': 'cemberut', '🙅': 'tidak', '💁': 'minta', '🤦': 'aduh', '🤷': 'tidak tahu', '🤞': 'menyilangkan jari', '📉': 'grafik turun', '⛔': 'tidak boleh masuk', '✖': 'tidak', '❌': 'tidak', '❎': 'tidak', '👌': 'ok', '👊': 'kepal tangan', '🤘': 'tanda tanduk', '😍': 'senyum cinta', '😊': 'senyum', '👍': 'baik','👍👍👍':'sangat baik', '😹': 'air mata bahagia', '👏': 'tepuk tangan', '😘': 'cium', '😂': 'air mata bahagia', '🙏': 'melipat tangan', '✊': 'jaya', '🌟': 'bintang sinar', '😁': 'seringai mata senyum', '😀': 'seringai', '💘': 'panah hati', '✔': 'tanda cek', '🤗': 'peluk', '😚': 'cium', '❤': 'cinta', '🙋': 'angkat tangan', '🙌': 'acung tangan', '🤣': 'tertawa terbahak', '😆': 'tertaw', '😅': 'senyum mulut terbuka keringat dingin', '😄': 'senyum mulut terbuka', '😎': 'senyum dengan  kacamata', '🏆': 'trofi', '✌': 'tangan menang', '😃': 'senyum', '😉': 'kedip mata', '😗': 'cium', '😙': 'cium', '☺': 'senyum', '🙂': 'agak senyum', '😇': 'senyum halo', '🤠': 'koboi', '🤡': 'badut', '😺': 'senyum', '😸': 'seringai', '😻': 'senyum mata cinta', '😼': 'cengir', '😽': 'cium', '✈': 'pesawat', '👼': 'malaikat', '💆': 'pijat', '🚶': 'jalan', '💃': 'tari', '👭': 'pegang tangan', '💏': 'cium', '💑': 'pasangan dengan hati', '💪': 'bisep', '🖐': 'melambai', '🤝': 'salam', '💋': 'bibir', '💞': 'hati berputar', '💝': 'hati', '💎': 'berlian', '🐥': 'anak ayam', '💐': 'buket', '🌹': 'Rose', '🌛': 'seperempat bulan pertama', '🌜': 'seperempat bulan terakhir', '🌝': 'bulan penuh tersenyum', '🌞': 'matahari', '⭐': 'bintang', '🌈': 'pelangi', '🎀': 'pita', '🎁': 'hadiah', '💡': 'bohlam', '📈': 'grafik naik', '💯': 'seratus', '🆗': 'ok', '•':'', '🥲':''}
with open('../src/slang_words.txt') as f: data = f.read()
slang_words = json.loads(data)
stopwords = stopwords_indo

# define preprocessing function
def preprocess_text(text: str, remove_stopwords: bool, stem: bool) -> str:
    """This utility function sanitizes a string by:
    - transforming in lowercase
    - converting emoji to description
    - converting slang words to standard words
    - removing links
    - removing special characters
    - removing numbers
    - removing stopwords
    - removing excessive whitespaces
    - stemming
    Args:
        text (str): the input text you want to clean
        remove_stopwords (bool): whether or not to remove stopwords
    Returns:
        str: the cleaned text
    """
    
    # lower text and stripped of whitespaces
    text = text.lower().strip()
        
    # convert emoji to description
    text_split = text.split(" ")
    for i in text_split:
        if i in list(emoji_converter_indo.keys()):
            text_split[text_split.index(i)] = emoji_converter_indo[i]
        else:
            pass
    text = " ".join(text_split)
    
    # convert slang to standard words
    text_split = text.split(" ")
    for i in text_split:
        if i in list(slang_words.keys()):
            text_split[text_split.index(i)] = slang_words[i]
        else:
            pass
    text = " ".join(text_split)
        
    # remove links
    text = re.sub(r"http\S+", "", text)
    
    # remove special chars and numbers
    text = re.sub("[^A-Za-z]+", " ", text)
    
    # remove stopwords and particular words
    if remove_stopwords:
        # 1. tokenize
        tokens = nltk.word_tokenize(text)
        # 2. check if stopword
        tokens = [w for w in tokens if not w.lower() in stopwords]
        # 3. join back together
        text = " ".join(tokens)
    
    # remove whitespaces
    text = text.strip()
    
    # stemming
    if stem:
        text = stemmer.stem(text)

    return text