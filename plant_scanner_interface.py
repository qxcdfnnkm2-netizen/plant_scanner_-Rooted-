import os
import json
import base64
import inspect
import re

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# ==================================================
# FILE PATHS
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

BACKGROUND_PATH = os.path.join(BASE_DIR, "hintergrund.jpg")
MODEL_PATH = os.path.join(BASE_DIR, "plant_scanner_rooted.keras")
CLASS_PATH = os.path.join(BASE_DIR, "plant_scanner.json")
INFO_BACKGROUND_PATH = os.path.join(BASE_DIR, "info_background.jpg")


# ==================================================
# PLANT ICONS (COLLECTION)
# ==================================================

PLANT_ICONS = {
    "birke_(betula_pendula)": "birke_icon.png",
    "bohne_(faba)": "bohne_icon.png",
    "eiche_(quercus)": "eiche_icon.png",
    "eibe_(taxus_baccata)": "eibe_icon.png",
    "esche_(fraxinus_excelsior)": "esche_icon.png",
    "gemeiner_schneeball_(viburnum_opulus)": "schneeball_icon.png",
    "gerste_(hordeum_vulgare)": "gerste_icon.png",
    "hasel_(corylus_avellana)": "hasel_icon.png",
    "kirschbluete_(prunus_spp)": "kirschbluete_icon.png",
    "kuerbis_(cucurbita_spp)": "kuerbis_icon.png",
    "linde_(tilla_cordata)": "linde_icon.png",
    "mais_(zea_mays)": "mais_icon.png",
    "waldkierfer_(prinus_sylvestris)": "waldkiefer_icon.png",
    "weissdorn_(crataegus_monogyna)": "weissdorn_icon.png",
}


def load_icon_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def get_plant_icon_html(plant):
    icon_file = PLANT_ICONS.get(plant)
    icon_path = os.path.join(ASSETS_DIR, icon_file) if icon_file else None
    if icon_path and os.path.exists(icon_path):
        encoded = load_icon_base64(icon_path)
        icon = f'<img class="collection-icon-img" src="data:image/png;base64,{encoded}" />'
    else:
        icon = '<div class="collection-icon-emoji">🌱</div>'
    return f'<div class="collection-icon-wrap">{icon}</div>'


def get_popup_icon_html(plant):
    icon_file = PLANT_ICONS.get(plant)
    icon_path = os.path.join(ASSETS_DIR, icon_file) if icon_file else None
    if icon_path and os.path.exists(icon_path):
        encoded = load_icon_base64(icon_path)
        icon = f'<img class="popup-icon-img" src="data:image/png;base64,{encoded}" />'
    else:
        icon = '<div class="popup-icon-emoji">🌱</div>'
    return f'<div class="popup-icon-frame">{icon}</div>'


# ==================================================
# BADGE ICONS
# ==================================================

BADGE_ICONS = {
    "three_sisters": "icon_three_sisters.png",
    "sacred_sites": "icon_sacred_sites.png",
    "noble_trees": "icon_noble_trees.png",
    "natures_pharmacy": "icon_natures_pharmacy.png",
}


def get_badge_card_icon_html(badge_key):
    icon_file = BADGE_ICONS.get(badge_key)
    icon_path = os.path.join(ASSETS_DIR, icon_file) if icon_file else None
    if icon_path and os.path.exists(icon_path):
        encoded = load_icon_base64(icon_path)
        icon = f'<img class="badge-icon-img" src="data:image/png;base64,{encoded}" />'
    else:
        icon = '<div class="badge-icon">🏅</div>'
    return f'<div class="badge-icon-wrap">{icon}</div>'


def get_badge_popup_icon_html(badge_key):
    icon_file = BADGE_ICONS.get(badge_key)
    icon_path = os.path.join(ASSETS_DIR, icon_file) if icon_file else None
    if icon_path and os.path.exists(icon_path):
        encoded = load_icon_base64(icon_path)
        icon = f'<img class="popup-icon-img" src="data:image/png;base64,{encoded}" />'
    else:
        icon = '<div class="popup-icon-emoji">🏅</div>'
    return f'<div class="popup-icon-frame">{icon}</div>'


COLLECTION_CARD_TONES = ["#eef3df", "#e6efd6", "#f1f5e4", "#e9f2dd"]


def get_collection_card_tone(index):
    return COLLECTION_CARD_TONES[index % len(COLLECTION_CARD_TONES)]


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Plant Scanner",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==================================================
# BACKGROUND
# ==================================================

def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image:
                linear-gradient(rgba(255,255,255,0.35), rgba(255,255,255,0.35)),
                url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """, unsafe_allow_html=True)


set_background(BACKGROUND_PATH)


# ==================================================
# GENERAL CSS
# ==================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Lora:wght@400;500&display=swap');

.block-container { max-width: 1100px; padding-top: 1rem; padding-bottom: 3rem; }
[data-testid="stSidebar"] { display: none; }
[data-testid="stHeader"] {
    background: #1f2b16 !important;
    height: 2.4rem !important;
}
[data-testid="stHeader"] * {
    color: #9dc98a !important;
    fill: #9dc98a !important;
}
[data-testid="stToolbar"] { top: 0.3rem !important; }

html, body, p, span, div {
    font-family: 'Lora', serif !important;
    color: #2f3a1f !important;
}
[data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded' !important;
}
h1, h2, h3, h4,
h1 *, h2 *, h3 *, h4 * {
    font-family: 'Playfair Display', serif !important;
    color: #56224b !important;
}

/* MAIN CONTENT */
.main-card {
    background: rgba(255,255,255,0.72);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(120,120,120,0.18);
    border-radius: 28px;
    padding: 30px;
    margin-bottom: 22px;
}

/* COLLECTION */
.collection-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 18px;
}
.collection-card {
    border-radius: 24px;
    text-align: center;
    border: 1px solid rgba(120,120,120,0.20);
    padding: 24px 16px;
}
.collection-icon-wrap {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-bottom: 14px;
}
.collection-icon-img {
    width: 160px;
    height: 160px;
    object-fit: contain;
    display: block;
}
.collection-icon-emoji { font-size: 70px; }
.collection-card-name {
    font-family: 'Playfair Display', serif !important;
    color: #56224b !important;
    font-size: 18px;
    font-weight: 700;
}

/* POPUP ICON — 1.5× (138/96) */
.popup-icon-frame {
    width: 160px;
    height: 160px;
    min-width: 160px;
    border-radius: 50%;
    background: #cfe0ba;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.popup-icon-img {
    width: 133px;
    height: 133px;
    object-fit: contain;
    display: block;
}
.popup-icon-emoji { font-size: 63px; }

/* POPUP HEADER ROW */
.popup-header {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 18px;
    margin-bottom: 24px;
    flex-wrap: wrap;
}
.popup-header-text { text-align: center; }

/* BADGES */
.badge-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 18px;
}
.badge-card {
    min-height: 230px;
    border-radius: 26px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    padding: 18px;
    box-sizing: border-box;
}
.badge-unlocked {
    background: rgba(255,255,255,0.84);
    border: 2px solid rgba(120,120,120,0.22);
}
.badge-locked {
    background: rgba(160,160,160,0.20);
    border: 2px solid rgba(120,120,120,0.18);
}
.badge-icon-wrap {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-bottom: 10px;
}
.badge-icon-img {
    width: 180px;
    height: 180px;
    object-fit: contain;
    display: block;
    border-radius: 50%;
}
.badge-icon  { font-size: 68px; }
.badge-title {
    font-family: 'Playfair Display', serif !important;
    color: #56224b !important;
    font-size: 17px;
    font-weight: 700;
    margin-top: 8px;
}
.badge-text { font-size: 13px; margin-top: 6px; }
.badge-dot  { font-size: 72px; color: #999999; }

/* ==================================================
   TABLET  (≤ 900px)
================================================== */
@media (max-width: 900px) {
    .block-container { padding-left: 1.2rem; padding-right: 1.2rem; }
    .collection-grid { grid-template-columns: repeat(2, 1fr); }
    .badge-grid      { grid-template-columns: repeat(2, 1fr); }
}

/* ==================================================
   PHONE  (≤ 600px)
================================================== */
@media (max-width: 600px) {

    .block-container {
        padding-left: 0.8rem;
        padding-right: 0.8rem;
        padding-top: 0.5rem;
    }
    .main-card { padding: 20px; border-radius: 22px; }

    h1 { font-size: 2rem !important; }

    /* Collection */
    .collection-grid    { grid-template-columns: 1fr; }
    .collection-icon-img { width: 90px; height: 90px; }

    /* Badges */
    .badge-grid     { grid-template-columns: repeat(2, 1fr); gap: 10px; }
    .badge-card     { min-height: 150px; padding: 10px; }
    .badge-icon-img { width: 64px; height: 64px; }
    .badge-icon     { font-size: 44px; }
    .badge-title    { font-size: 13px; }
    .badge-text     { font-size: 11px; }
    .badge-dot      { font-size: 52px; }

    /* Popup icon */
    .popup-icon-frame {
        width: 90px;
        height: 90px;
        min-width: 90px;
    }
    .popup-icon-img   { width: 62px; height: 62px; }
    .popup-icon-emoji { font-size: 42px; }

    .popup-header { gap: 10px; }
}

/* ==================================================
   iOS Safari fix: background-attachment:fixed
   broken on iOS touch devices
================================================== */
@media (max-width: 900px) and (hover: none) and (pointer: coarse) {
    .stApp {
        background-attachment: scroll !important;
    }
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_plant_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_plant_model()


# ==================================================
# LOAD CLASS NAMES
# ==================================================

with open(CLASS_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)


# ==================================================
# DISPLAY NAMES
# ==================================================

pretty_names = {
    "birke_(betula_pendula)": "Silver Birch (Betula pendula)",
    "bohne_(faba)": "Broad Bean (Vicia faba)",
    "eibe_(taxus_baccata)": "European Yew (Taxus baccata)",
    "eiche_(quercus)": "Oak (Quercus)",
    "esche_(fraxinus_excelsior)": "European Ash (Fraxinus excelsior)",
    "gemeiner_schneeball_(viburnum_opulus)": "Guelder Rose (Viburnum opulus)",
    "gerste_(hordeum_vulgare)": "Barley (Hordeum vulgare)",
    "hasel_(corylus_avellana)": "Common Hazel (Corylus avellana)",
    "kirschbluete_(prunus_spp)": "Cherry Blossom (Prunus spp.)",
    "kuerbis_(cucurbita_spp)": "Pumpkin / Squash (Cucurbita spp.)",
    "linde_(tilla_cordata)": "Small-leaved Lime (Tilia cordata)",
    "mais_(zea_mays)": "Maize (Zea mays)",
    "waldkierfer_(prinus_sylvestris)": "Scots Pine (Pinus sylvestris)",
    "weissdorn_(crataegus_monogyna)": "Common Hawthorn (Crataegus monogyna)",
}


# ==================================================
# PLANT INFORMATION
# ==================================================

plant_information = {
    "birke_(betula_pendula)": """
Oh, how lovely! You've found yourself a birch tree.

For the Sámi People in Scandinavia, the birch is part of a much larger relationship between people, reindeer, and the land. Birch forests structure seasonal use by shaping reindeer grazing and guiding their movement across the landscape throughout the year. In Estonia, birch is counted among the documented sacred tree species, and birch twigs remain part of sauna traditions and embodied family knowledge passed down through generations [3].

In Old Norse mythology, the birch was sacred to the goddess Freya, goddess of fertility, spring, luck, and love. Some even trace the name „birch" back to the old Irish goddess Brigid, „the bright one" or „the radiant one". In Russia, the birch has been beloved since ancient times. It is one of the most common trees, appearing throughout songs, poems, and paintings as a beloved symbol of the land. The Slavs, too, considered the birch a sacred tree, connecting it with spring, purity, and natural, feminine beauty. Above all they understood it to be a symbol of new beginnings: since it is the first native tree to show green each spring, it became tradition for the maypole to be a young birch, and for a newborn's cradle to be built from birch wood [1], [2].

Among the Celts, Germans, and Scandinavians, the birch held a fixed place in local custom. In the Celtic „druid baptism", for instance, a teacher would mark this rite of passage with a light tap of a dew-moistened birch twig. In parts of Central Europe, a increasingly forgotten custom called the „Liebesmaien" saw young men place small, decorated birch trees at the door of the woman they loved on the night before the first of May, as a declaration of love [1].

The birch also carried its share of magic and myth. It was said that witches rode brooms made of birch wood on Walpurgis Night, using birch twigs to ward off spells. As the very first tree to turn green in spring, the birch became a lasting symbol of awakening and rebirth, a sacred tree representing the virgin goddess of spring's fertility festivals [1], [2].

You can now see that the birch has quietly accompanied entire peoples through the changing seasons, by guiding herders and their reindeer, welcoming newborns and lovers, and heralding, again and again, the return of spring.
""",
    "bohne_(faba)": """
Oh, how lovely! You've found yourself some beans. For centuries and millennia, they have served us as food, providing vital nutrition to many cultures. No wonder there are many different relationships that humans have developed toward beans!

In native American Hopi culture, they are seen as rainmakers from the gods and used in Powamuya ceremonies. These ceremonies are held in February and call the Katsinam to come help the Hopi prepare for the growing season. The Katsinam are hundreds of different spirits, that bring blessings to the Hopi people [4].

Another central event during Powamuya is the planting of beans. In the Kiva, a special circular underground room for rituals, between fifty and a hundred seeds are planted in a bucket filled with earth. The people care for the beans cautiously and keep a fire going day and night to help them grow. On the sixteenth day there is a public ceremony where the Katsinam dance and gift the people mature bean sprouts, dolls, decorative plates and many more gifts [4].

On the other side of the world, beans were at the center of a famous religious tabu in ancient Greece. There is much discussion over what actually happened. According to some accounts Pythagoras himself forbade the consumption of beans as part of the Pythagorean dietary restrictions, as they were connected to Hades and thus possibly connected to metempsychosis, which is the reincarnation of the dead or the transmigration of the soul. Other ancient scholars thoroughly disagreed with such a notion, and reported he liked beans best of all vegetables [5].

We may never know Pythagoras' true stance on beans, but it seems he was no way near impartial to them. Whichever way it certainly sparked a lot of debate and possibly shaped the ancient Greeks' way of life.

I hope you can see that even though beans have held incredibly different meaning for different cultures over time, our relationship with them far surpasses mere utility.
""",
    "eibe_(taxus_baccata)": """
Careful! You've encountered a yew tree on your journey.

In Ireland, the yew has been connected to rituals and remembrance for a long time. In ancient Ireland, the deep red heartwood of the yew was carved into shrines for holy books and relics. The Irish County Mayo even owes its very name to the tree: Maigh Eo, meaning Plain of the Yew Trees. Nine yews still appear on the county crest, one for each of its baronies, and its red and green colors symbolize the tree's bright leaves and red berries. Its branches were once used on Palm Sunday, taken home after the church ceremony and worn or hung in the house for good luck and protection. Its wood was burned to make the ash used on Ash Wednesday. Yews have grown rarer across the Irish landscape since, as farmers came to fear their toxic berries would poison livestock, and today they are found mainly in parks and graveyards [6].

That association with graveyards runs deep across various cultures. Old yews are often left untouched in cemeteries, especially in Ireland, England, and Brittany, out of a belief that a yew root grows from every buried body. The tree's dark leaves and poisonous berries have long tied it to death. Even Shakespeare knew this connection: he describes the custom of tucking yew twigs into burial shrouds, and in the final scene of Romeo and Juliet, Balthasar speaks of falling asleep beneath a yew and dreaming of a fatal duel. This was a nod to further old beliefs about the tree, since sleeping beneath a yew was thought to cause hallucinations. This was later found to have some basis in fact, as the yew releases a gaseous toxin on warm days [6], [7].

The yew's dangerous reputation even reaches back to Roman and Celtic times. Julius Caesar wrote that Germanic tribes had poisoned their arrows with yew sap in an attempt to kill him, and that a leader of the Eburones took his own life with yew poison rather than be captured by the Romans. The Romans furthermore saw the yew as a tree that guarded the underworld. Similarly, the ancient Greeks regarded it as a gateway to the underworld and a guardian of the soul. For the Celts, the yew was also thought to keep watch between the world of the dead and the living. Their druids held it sacred for its connection to eternity and used it for wands and divining rods and as protection against evil spirits [7].

In more recent history, the yew has taken on an entirely different role: in the early 1980s, researchers discovered that its bark yields paclitaxel, a highly effective cancer medication. Its poison had, in earlier times, also been dangerously used to induce abortions, at times poisoning the women themselves [6].

You see, few trees hold such a striking balance of opposites as the yew. It is both a keeper of graves and a guardian against evil, a common poison in the past, yet possibly a source of healing in the future.
""",
    "eiche_(quercus)": """
How magnificent! You've encountered an oak.

The oak is among the most sacred of trees since ancient times. It is astonishing how similar its symbolism is across many different Peoples. In ancient Greece, for example, the oak was the tree of Zeus himself. At the oracle of the city Dodona, three priestesses in white robes received Zeus voice through listening to the rustling of a sacred oaks' leaves. The forest nymphs of Greek mythology, called the „dryads", even took their name from „drys", the Greek word for oak. Zeus' Roman counterpart, Jupiter, was likewise a god of thunder and father of the gods. His tree, too, was the oak. Together with the palm, it was a symbol of the golden age in ancient Rome [8].

Even among the Celts, the oak belonged to Taranis, the ruler of the sky and God of weather. Even until today, the language itself preserves the oaks sacred status. The Celtic word for priest, „druid", is thought to derive from „duir", meaning oak. Even the words for door and gate trace back to this same root. Cutting down a sacred oak grove without permission was punishable by death. The Irish also connected the oak with their own weather deity, called „Dagda". Similarly, the Slavs dedicated it to „Perun", the highest god of the panteon and the god of the sky, weather and war. His name, too, comes from the old word for oak [8].

But the similarities don't end here. In Norse and Germanic tradition, the oak was sacred to the thunder god Thor, or „Donar" as they called him. The oak was thought to hold both masculine and feminine qualities. On the one hand it represented strength, glory, and pride through Thor, on the other nurture through the ancient mother figure „Ana", who was said to feed both humans and animals with its acorns. Legends even said that the first human was born from an oak. The Germanic peoples worshipped their gods in the forest, and sacred oak groves were strictly forbidden to outsiders. Warriors hung their battle trophies on these trees and were crowned with oak leaves by the priests [8].

Sadly, Christianization did not spare these sacred oaks. In the year 723, the missionaries felled what was probably the most significant of these sacred trees, the Donar Oak near Geismar in Hesse. Countless other sacred groves across the land soon followed. But even centuries after this, the respect and admiration for the oak lingered on within the common people. Thus, under the Church the oak found new meaning as a symbol for Christ and Virgin Mary [8].

Centuries later, the oak took on a completely different kind of significance. In the eighteenth century it became the symbolic tree of the German people, that stood for freedom, pride, strength, and endurance. It appeared on the Iron Cross as a mark of courage as early as 1813, and still adorns German cent coins today [8].

I hope you can see just how deep the connection we share with oaks truly is, as it has carried extraordinarily profound meanings across thousands of years. It has been a home for gods and spirits, a source of life and nourishment, and a steady partner all along.
""",
    "esche_(fraxinus_excelsior)": """
Wonderful! You've encountered an ash tree.

In Ireland, the ash is a native species long tied to wisdom, kingship, and mystical knowledge. Its wood was once burned to banish the devil, and a staff carved from ash was believed to protect its bearer against any evil. Ash trees are often found growing beside holy wells, the sap of a young ash trees was used to cure earaches, while the smoke of smoldering ash twigs was believed to cure ringworm [9].

In Norse and Germanic mythology, the ash takes on an even greater role, as the world tree Yggdrasil. With its three roots, Yggdrasil connects nine different worlds spread across three levels. These worlds are home to elves, dwarves, giants, and even gods, with humankind living at the center, in Midgard. Should this tree ever begin to wither, it would bring about the end of all worlds. A dragon gnaws at one of its roots, trying to poison it, while an eagle lives in its crown as its eternal rival, and a squirrel constantly stirs unrest between the two, keeping the endless struggle between good and evil alive [10].

The ash appears at the very origin of humankind in other cultures as well. As early as the seventh century BC, the Greek poet Hesiod wrote about how Zeus shaped the third generation of humans from ash wood. In Germanic mythology, Ask (ash) and Embla (elm) are the first two humans, formed by the gods out of two pieces of driftwood. They received their souls from Odin, the warmth of life from Hönir, and their appearance from Lodur. Among the Algonquin people of North America, it is told that the creator of the world brought forth man and woman by shooting an arrow into an ash tree [10].

Ash wood also carries its own legend of heroism: the most famous spear of ash, that of the centaur Chiron, was said to be cut from a sacred ash growing on Mount Pelion above his cave. Achilles carried this spear into battle at Troy, and it was with this very weapon that he killed Hector [10].

Beyond myth, the ash was also long believed to hold power over water. Druids used its wood for rain magic and to hold back floods, fishermen built parts of their boats from ash to protect them from capsizing and its bark and leaves found their way into folk remedies for fevers and wounds for centuries afterward [9], [10].

You see, the ash is far more than a random tree of the forest. To many cultures, it has been a protector in the present, and the very origin of people in the past.
""",
    "gemeiner_schneeball_(viburnum_opulus)": """
What a special find! You've come across a guelder rose, known in Ukraine as kalyna.

Few plants carry a bond with a people as deep as kalyna does with Ukraine. It is seen as a companion to people, from birth to death. Kalyna symbolizes motherhood in a very literal way: the bush represents the mother, while its blossoms and berries represent her children. Beyond that, it embodies one's native land, one's father's home, and everything dear to the heart, which is why it has come to be regarded as a national treasure, loved and revered throughout the country. The ancient Slavs went further still, seeing in kalyna a symbol of rebirth and the eternity of life. Like the oak, it stands for beauty and strength, but of an entirely tender, extraordinary kind [11].

I hope you can see that this small shrub carries something far larger than its size suggests — an entire people's sense of home, motherhood, and belonging, held within its blossoms and berries.
""",
    "gerste_(hordeum_vulgare)": """
Oh, how wonderful! You've come across some barley.

In Tibetan communities, barley is far more than just food. It is eaten during traditional ceremonies and worship rituals and takes center stage at the most important festivals of the year. Many Tibetans believe that barley was a gift passed down from an ancestral god. The different local varieties of barley are carefully conserved as an act of respect toward one's own ancestors, who preserved these same seeds from generation to generation. Barley also plays a central role in weddings. Its grains are scattered in front of the homes of both the bride and groom to form a lucky painting. Furthermore, a bag of barley is placed for the bride to stand upon as she leaves her family and again as she arrives at her new home. This ceremony is understood as a symbol for a rich and abundant life for the newlyweds [12].

Among the Shuhi people of the eastern Himalaya, „tsampa", a roasted barley flour, and barley wine are equally significant in daily worship. Each morning, the Shuhi burn incense on the flat roofs of their homes, pouring a spoonful of tsampa mixed with marigold, and a sip of barley wine, over the offering as food for the gods. Barley also anchors two harvest festivals held every year, which entail an elaborate exchange of food between households. At the beginning of the barley harvest on the first and second day of May, renewal is the central motto. Any remaining bottle of last year's barley wine has to be finished before the new wine can be served, and old tsampa is replaced with new. In people's homes, new wine and tsampa, mixed with marigold, are burnt as an offering, to thank the gods for the harvest just gathered and to ask them for large crops in the future [13].

I hope you can see that barley is far more than a crop harvested for the table. For multiple communities, it is a gift entrusted by both ancestors and gods, that gets renewed each year through ritual, gratitude, and the hope for what the next harvest will bring.
""",
    "hasel_(corylus_avellana)": """
Marvelous! You have found a Hazel… an ancient keeper of wisdom.

A native species of Ireland, the Hazel is bound up with wisdom, kingship, and mystical knowledge, its roots reaching back to the very first hunter-gatherer peoples who settled this land. At the heart of its legend stand the nine hazels of wisdom, growing at the Well of Segais, the source of the Boyne or the Shannon River. Their leaves, blossoms, and nuts all fell into the water at once, and the salmon living there fed upon them, gaining a red spot of knowledge with every nut they swallowed. When the hero Fionn Mac Cumhaill later tasted this salmon of knowledge, its wisdom passed into him as well. This plant was held close for millennia by Irish myth, folklore and poetry [14].

Beyond the legend of the nine hazels of wisdom, this plant also is known to carry luck and protection. Finding a double hazelnut, also called a St. John's Nut, is considered a stroke of great fortune, and carrying a hazel stick while traveling at night is said to guard against evil spirits and mischievous fairies [14].

Alongside its mythical meaning, the hazel has found practical use for thousands of years. As early as the Mesolithic period, it was used to make fish traps, and by the Neolithic it was among the most popular firewoods. For centuries, hazel brushwood formed trackways across Irish bogs, while hazelnut shells recovered from ancient sites reveal how large of a part hazelnut once played in the diet of early communities. In early Medieval Ireland, the hazel was honored as one of the seven nobles of the wood, so valued that early Irish law texts imposed a hefty fine for damaging one [14], [15].

Now that you have encountered this ancient companion on your way, may its wisdom light your path and its quiet protection walk beside you.
""",
    "kirschbluete_(prunus_spp)": """
Beautiful! You've come across a cherry tree.

In Japan, the cherry blossom, or sakura, has long represented the nature of life itself. Not only is the beauty of the flowers short and sweet, but the trees themselves are relatively short-lived too. Sakuras hold contradictory meaning, as they symbolize both birth and death, beauty and violence at once. They have long been a central motif in the Japanese worship of nature, but have historically also signified the brief, colorful life of a samurai, and during the Second World War, kamikaze pilots marked their planes with sakura emblems before their final missions [16], [17].

Many kodama, or tree spirits, are said to dwell in cherry trees. One of the most famous tales tells of the Uba-zakura, the Milk Nurse Cherry Tree, said to blossom each year on the anniversary of the death of a devoted nurse. She is said to have given up her own soul and life to save a child in her care, living on in the form of a Sakura ever since. Another old story tells of a lonely, aging samurai whose only comfort in his final years was the ancient cherry tree in his garden. When the tree itself died one summer, the grieving samurai took his own life beneath it the following January. His spirit is said to have entered the tree, making it bloom again each year on that same day ever since [16].

Furthermore, there is a ritual of hanami, or „flower viewing", that reaches back to ancient times, when farmers would pray, make offerings, and hold feasts beneath the blossoming trees in hopes of a bountiful harvest. Today, hanami remains a beloved annual tradition, where families and friends gather each spring for picnics beneath the blossoms. In Japanese folk religion, cherry trees were also thought to be dwelling places for mountain deities, who transformed each spring into the gods of the rice paddies, drawing people up into the mountains to worship the blossoming trees [17].

Elsewhere, cherry trees have also carried their own connection to the spirits. They were believed to be home to forest and tree spirits, and it was said that under a full moon, when the bark of a cherry tree shimmers silver, one might catch sight of elves dancing around its trunk (though watching them was strictly forbidden and thought to invite misfortune). This belief likely reaches back to the ancient Greeks, who held the cherry tree sacred to the goddess Artemis. She was tied to death and the underworld and at the same time to fertility, birth, and joy [18], [19].

I hope you can now see that, whether in Japan or across Europe, the cherry tree has always carried the same paradox: a symbol of fleeting beauty connected equally close with birth, love, death, and the otherworldly.
""",
    "kuerbis_(cucurbita_spp)": """
Oh, great! You've found yourself a pumpkin, or one of its many relatives.

In Hopi and Zuni mythology, squash has its own guardian spirit, Patung, known as the Squash Kachina. Patung is said to have taught the Puebloan peoples how to plant corn before vanishing, and is remembered as a shapeshifter, able to become a badger, associated with healing, protection, and knowledge of roots and herbs. He is said to appear in the fall months, hiding in the shade among the harvest, and katsina dolls on Hopi land are often carved from dried gourds, pumpkins, or squash [20], [21].

Pumpkins also have a more recent, but no less magical, life in Irish and wider European tradition. Long before pumpkins themselves were used, people across Ireland and Britain carved lanterns out of turnips and mangel wurzels at Halloween time, giving them grotesque faces to ward off evil spirits or to represent the spirits themselves. This custom may reach back to ancient Celtic practices of venerating the heads of the dead for spiritual protection. As Halloween coincided with Samhain, a night when supernatural beings were believed to walk the earth, these carved lanterns were set on windowsills or carried about to frighten away what wandered outside [22].

As you can see, pumpkins and squash carry a relationship with people that goes far beyond the harvest table. They are a protective spirit in the fall and a guardian against the otherworld on the darkest night of the year.
""",
    "linde_(tilla_cordata)": """
You have found a linden tree! How wonderful!

For many centuries and millennia, linden trees have held a special significance for various different cultures and thus shared a deep relationship with many people.

To begin, I want to introduce you to lime trees as a companion of whole communities and villages. In the German culture, she has been understood as a symbol of home and belonging for centuries and millennia. The people sang about her in songs and paid tribute to her in poems, more so than for any other tree. This connection reaches back to times of ancient Germania, where lime trees were physically and metaphorically placed at the center of communities, to serve as their place for gathering. Similarly, in Scandinavian cultures, the lime tree was the most important of three protection trees, that served to protect people's homes and lands and were given sacrifices accordingly [23].

But lime trees have accompanied us in many more ways. They also still serve as places of remembrance and living monuments in Germany. For instance, trees like the „Goethelinde" or the „Freiheitsbäume" (freedom trees) planted after the German reunification invite people to remember and pay respect to important people and events that shaped German culture and history [23].

Lime trees have also provided places of spiritual and religious connection across many cultures. The seers of the Scythes, an indogermanic nomad people, used the bark of lime trees to make predictions about the future. Interestingly though, almost all spiritual connections humans shared with lime trees, across wide timelines and locations, relate back to deities of love. In ancient Germania, Lime trees were deeply connected to the goddess Freya, who symbolized fertility, love and good luck. They were deemed to be a home of good spirits. Slavic cultures also deemed lime trees to be holy places, where they worshipped their own goddess of love. And even in ancient Greece lime trees were seen to have a special connection to the goddess of love Aphrodite and again understood to be places where spirits reside [24], [25].

I hope you now see that they are not just plants, separate from us, but woven into the very fabric of people's cultures, histories and identities.

What a gift to share our history with such a steady companion!
""",
    "mais_(zea_mays)": """
Oh, how exciting! You've come across some maize, or corn.

Although it has been the cornerstone of many culture's diets for thousands of years, corn has meant far more to people than just something to eat. Across continents and centuries people have understood it to be a gift from the gods, a family member, and even the very substance from which humans were made.

In modern day Arizona, the Hopi people see corn as the thread that knits everything together. According to some legends, when people first emerged into this world from the underworld, the guardian of the Earth allowed each tribe to choose an ear of corn, and with it, a way of life. While other tribes took the large ears that promised short, easy lives, the Hopi took longer to choose and were left with only a short ear of blue corn. This blue corn brought a long and difficult life, but also the strength to survive hard times. To this day, corn nourishes not only the Hopi People but also their traditional gods, katsina spirits. The Hopi are born, live, and die with the blessings of cornmeal in their ceremonies [29].

Similarly, the Myaamia people of Oklahoma and Ohio have their own deep bond with corn, known to them as miincipi. The corn grown today descends from only two dried cobs that survived generations of hardship. Bringing miincipi back to their fields and festivals has become an act of cultural reclamation, connecting language, lunar calendar traditions, and the memory of ancestors who once tended it [26].

In Mesoamerica, the Maya honored a Maize God who is deeply tied to the cycle of life and death, since corn is planted and harvested anew within a single year. He can appear in different forms depending on the stage of the corn's growth. According to one Maya creation myth, the first humans themselves were shaped from dough made of sacred corn [27], [28].

But corn has played an important cultural role on other continents too. In western Kenya, among the Luo people, corn seed passed down from one's ancestors is treated as family seed, carrying with it the presence and authority of elders long gone. Choosing to save and replant this seed, rather than buying seeds from the market, is understood as an act of loyalty to one's own history and ancestors [30].

I hope you can see that corn is not simply a crop that feeds us, but a living inheritance. It carries stories of survival, identity, and is fundamental in the relationship between very different cultures and their respective ancestors and gods.
""",
    "waldkierfer_(prinus_sylvestris)": """
What a find! You've come across a Scots pine.

Since ancient times, the pine has carried a quiet but persistent symbolism across many cultures. Its cone, packed with forty to fifty seeds, has been seen as a sign of fertility and abundance for a long time. The pine has also stood for longevity, endurance, and resurrection — in parts of Eastern Europe, it is still said that the nails used to crucify Jesus were carved from pine wood [31].

Nowhere has the pine held greater standing than in Japan, where it ranks among the most beloved garden trees alongside the plum and the cherry, and is revered in old belief as a seat of the gods. To this day, it is traditional to place pine branches on either side of the front door at New Year, decorating an otherwise bare entrance — a custom that gives the pine roughly the same treasured place in Japanese tradition as the fir or spruce holds for us at Christmas [31].

In medieval Central Europe, the pine served a far more practical kind of magic. Its resin-rich wood was cut into finger-thick splinters called Kienspan, dried and dipped in resin or pitch so that they would burn slowly, providing one of the most important sources of light in caves, huts, tents, and houses well into the early twentieth century, especially among poorer households who could not afford wax candles. Even its needles were put to use: soaked in warm water until their tough outer casing burst, they could be turned into a soft, wool-like material used to stuff pillows, cushions, and mattresses [31].

The pine has also long had its place in the customs of hunters. Broken twigs, known as Bruchzeichen, serve as a traditional form of silent communication between hunters in the field, and the most significant of these, the Erlegerbruch, is presented to a hunter by the hunt leader as a mark of recognition for a successful hunt — a tradition as old as hunting itself [32].

I hope you can see that, even where the pine plays a lesser role in myth and legend, it has quietly accompanied people through everyday life — as a source of light, warmth, comfort, and recognition, from Central European households to Japanese doorways.
""",
    "weissdorn_(crataegus_monogyna)": """
Beware! You have encountered a Hawthorne on your journey!

If you're lucky you are catching them draped in beautiful white petals in spring, or maybe they already bear the striking red berries they carry in autumn. Whatever dress you're presented, Hawthornes have a significance far greater than their beauty. They actually play a major role in Irish mythology [33].

Hawthornes are understood to be magical fairy trees. Though this might sound pleasant at first, they are actually widely respected or even feared. People actively avoid disturbing these trees and the „little people", as they understand them to be connected to otherworldly places and beings. More so, there is a widespread belief, that Hawthorne blossoms are unlucky and bringing them into one's home could evoke death [33].

There is even a belief that Hawthorne has its own temper, which is why you should never hit someone with a Hawthorne stick! Bringing such a stick into one's home would again bring about discontent and trouble [33].

Alongside these specific mythological connections, they also serve a more practical purpose to the Irish people: the bright white flowers of Hawthorne announce the arrival of summertime [33].

As you can see, the Hawthorne shares a deep complex relationship with its people. They are not to be underestimated, but to be treated with the respect they deserve. And who knows, if you look closely, you might even spot a fairy or two!
""",
}


# ==================================================
# BADGE DEFINITIONS
# ==================================================

BADGES = [
    {
        "key": "three_sisters",
        "title": "Three Sisters",
        "required_plants": [
            "mais_(zea_mays)",
            "bohne_(faba)",
            "kuerbis_(cucurbita_spp)",
        ],
        "text": """
Congratulations, you’ve unlocked new knowledge! As you have encountered corn, beans, and squash, it is now time to learn about their shared significance as the Three Sisters.

Long before they were understood as a clever planting technique, corn, beans, and squash were understood to be family. Across many Indigenous cultures of the Americas, from the Haudenosaunee to the Muscogee, Mandan, and Cherokee, these three plants were treated not as separate crops but as sisters, bound to one another, and to the people who grew them [34].

Among the Haudenosaunee, the three are known as Deohako, „the ones who sustain us", and are believed to hold both physical and spiritual life within them. One telling traces their origin back to Sky Woman, who first brought terrestrial land and life to earth. Her daughter died giving birth, and from her body the Three Sisters grew as a gift of agriculture passed down to her people. In another story, they appear as three sisters of very different natures: the youngest small and dressed in green, only able to crawl; the second in yellow, quick and fond of running; and the eldest, tall with long hair, standing watch over the other two. According to the story, the sisters wandered apart for a while, but they were reunited at harvest and vowed never to part again [34], [35].

That bond of care is what people still see reflected in the way the plants grow. The tall corn lends the bean something to climb; the bean quietly feeds the soil in return; and the squash spreads low at their feet, sheltering the ground beneath. To many who tend a Three Sisters garden today, this is read less as biology than as a teaching: a picture of what it looks like when different gifts are given freely and none is asked to stand alone. For many communities sitting down to eat corn, beans, and squash together is still a way of giving thanks for a harvest that was never earned through collaboration between plants and people [34], [35].

The Three Sisters were never only a way of growing food. They were, and still are, a way of understanding how to live alongside others: in reciprocity, patience, and with the trust that what is given will be returned.
""",
    },
    {
        "key": "sacred_sites",
        "title": "Sacred Sites",
        "required_plants": [
            "eiche_(quercus)",
            "waldkierfer_(prinus_sylvestris)",
            "linde_(tilla_cordata)",
        ],
        "text": """
Congratulations, you`ve unlocked new knowledge! As you have encountered oaks, pines, and lindens, it is now time to learn about their shared significance in the sacred sites of estonia!

Sacred natural sites are places tied to folk ritual and belief and often carry the identity of an entire local community within them. In general, sacred natural sites of indigenous peoples are the oldest protected areas of humankind. For centuries and millennia, humans have naturally connected their own spirituality with their role as stewards and partners of nature [36].

In Estonia alone, more than 800 sacred sites have been recorded. They are centered around groves of certain sacred trees like oak, pine, and linden, or around springs or sacred rocks. Some belong to a whole village, like a shared grove or healing spring, others are familial, a single tree tended and returned to by one family across generations. Strict customs have long protected the sacred groves and their trees. Branches are not to be broken there, the ground not to be dug and animals not to be herded through. Disturbing such a place is thought to invite illness, or worse, upon those responsible [36], [37].

One noteworthy tradition surrounds the so-called „cross trees": when a funeral procession passed a certain tree on its way to the cemetery, a cross was cut into its bark, both to remember the dead and to keep their soul tied to the tree rather than wandering as a restless ghost. Sacred springs, on the other hand, were treasured for their power to heal and refresh the soul. Water carried home from such a spring was trusted for healing and drinking, and it was common to offer a coin or a small scrap of silver into the water in return. At all these places, people came to pray, to heal, to ask a blessing for a coming marriage or the name of a newborn, to mark the turning of the year, or simply to leave a ribbon tied to a branch as a small offering [37], [38].

This sense of kinship with nature runs deep in the Estonian native belief of Maausk, which understands the land itself to be animate: not only people, but plants, animals, and the earth all carry a soul of their own. The Earth is honored as Maaema, a life-giving mother and the source of both the natural world and human culture alike, with people seen not as nature's rulers but simply as one part of it. Even today, this feeling has not faded. Many Estonians still describe their forests as something apart from ordinary life, close to what one might call the true church of the Estonian people, and continue to believe, as their ancestors did, that the trees around them carry a soul [36], [39].
""",
    },
    {
        "key": "noble_trees",
        "title": "Noble Trees",
        "required_plants": [
            "eiche_(quercus)",
            "hasel_(corylus_avellana)",
            "eibe_(taxus_baccata)",
            "esche_(fraxinus_excelsior)",
            "waldkierfer_(prinus_sylvestris)",
        ],
        "text": """
Congratulations, you’ve unlocked new knowledge! 
As you have encountered oak, hazel, yew, ash, and Scots pine, it is now time to learn about their shared status in one of the oldest tree laws in Europe.

In early medieval Ireland, long before written statutes existed anywhere else on the continent, judges known as Brehons upheld a legal code called Bretha Comaithchesa, or „the judgements of neighbourhood". One section of this code, now known as the Old Irish Tree List, sorted twenty-eight native trees and shrubs into four ranked classes. For each tree, it entailed a very detailed description of the fine and compensation owed for damaging it [40], [41].

At the very top of this list stood the airig fedo, the „nobles of the wood". They were comprised of oak, hazel, holly, yew, ash, Scots pine, and wild apple. To fell one of these seven without having the explicit right to do so was treated as one of the most serious offenses a person could commit. In the eyes of the law, it was comparable to killing a person of noble rank [40], [42].

Each tree was prized for a reason: the oak for its acorns and its timber, the hazel for its nuts, the yew for wood strong enough to make weapons. One rank below sat the aithig fedo, the „commoners of the wood", like alder, willow, elm, birch and hawthorn. Damaging one of these was still punished, though not as strongly [41], [42].

This civil law was passed down orally for generations, before Christian monks began recording it in writing in the seventh century. It actually remained in force for a thousand years, until the English colonial administration finally suppressed it in the seventeenth century. Even the earliest churches built in Ireland were sometimes called Dairthech, „oak church", a small sign of just how deeply trees were woven into everyday Irish life [43], [44], [45].

I hope you can see that the connection we share with these trees, that you've now come to know through myths and stories, runs even deeper. For at least a millennium, we saw and respected them as equals, protecting them in a legal order that treated their worth as seriously as a person's.
""",
    },
    {
        "key": "natures_pharmacy",
        "title": "Nature's Pharmacy",
        "required_plants": [
            "linde_(tilla_cordata)",
            "weissdorn_(crataegus_monogyna)",
            "birke_(betula_pendula)",
            "esche_(fraxinus_excelsior)",
            "mais_(zea_mays)",
        ],
        "text": """
Congratulations, you’ve unlocked new knowledge! 
As you have encountered linden, hawthorn, birch, ash, and corn, it is now time to learn how each has, in its own way, long served as medicine.

For centuries, a cup of linden flower tea has been one of Europe's most trusted home remedies, brewed for colds, fevers, and headaches. But above all, it was praised for its gentle, calming effect on a racing heart or a restless mind. From Germany to the Balkans, to this day it remains one of the continent's best-loved herbal teas and is sometimes even called the „nectar of kings" [46], [47].

The hawthorn tells an almost contradictory story. In Irish folklore it was a tree to fear, its blossoms unlucky, its branches best left untouched. Yet in Western herbal medicine, hawthorn earned the very opposite reputation. Here it is known simply as the „herb of the heart". For centuries it was used as a gentle tonic to strengthen a weakened heart by both Irish and Romani healers. Even today, it remains one of the most studied plant remedies for heart health [48], [49].

Each spring, from Belarus to Finland to northern China, people have tapped birch trees for their sap. This practice was already recorded in the tenth century, when the traveler Ahmad ibn Fadlan described peoples along the Volga River drinking its fermented version. Taken fresh each year at the very start of spring, birch sap was trusted to cleanse the body and support the kidneys, a seasonal tonic as much a ritual as a remedy [50].

In German-speaking Europe, the ash offered its own home pharmacy. Its bark was dried and brewed as a remedy for fever and once even considered nearly as effective as the far costlier cinchona bark. A tincture known as „Eschengeist" was rubbed onto sore muscles and aching joints, and its leaves were scattered around homes and stables in the belief that snakes feared them [51], [52].

Even the fine golden threads inside a corn husk, nowadays so often thrown away, have their own history as medicine. Native American nations dried and brewed corn silk into a tea. Similar practices were documented in traditional Chinese medicine as early as the Ming dynasty. In both cultures, the tea was valued as a gentle remedy for the kidneys and urinary tract [53].

You see, long before any modern pharmacy, people were already turning to the trees and plants around them for healing, steeping, tapping, and rubbing what grew nearby into remedies that, in some cases, medicine has only recently learned to explain.
""",
    },
]


# ==================================================
# BADGE REFERENCES
# ==================================================

BADGE_REFERENCES = [
    # Three Sisters
    {"badge": "three_sisters", "ref_id": "R1", "source": 'Wikipedia Contributors, "Three Sisters (agriculture)," Wikipedia. [Online]. Available: https://en.wikipedia.org/wiki/Three_Sisters_(agriculture) (accessed Aug. 21, 2026).'},
    {"badge": "three_sisters", "ref_id": "R2", "source": 'R. W. Kimmerer, Braiding Sweetgrass: Indigenous Wisdom, Scientific Knowledge and the Teachings of Plants. Minneapolis, MN: Milkweed Editions, 2013.'},
    # Sacred Sites
    {"badge": "sacred_sites", "ref_id": "R3", "source": 'A. Kaasik, "Conserving sacred natural sites in Estonia," in The Diversity of Sacred Lands in Europe: Proceedings of the Third Workshop of the Delos Initiative – Inari/Aanaar 2010, J.-M. Mallarach, T. Papayannis, and R. Väisänen, Eds. Gland, Switzerland: IUCN; Vantaa, Finland: Metsähallitus Natural Heritage Services, 2012, pp. 61–73.'},
    {"badge": "sacred_sites", "ref_id": "R4", "source": 'H. Valk, "Sacred Natural Places of Estonia: Regional Aspects," Folklore: Electronic Journal of Folklore, vol. 42, pp. 45–66, 2009, doi: 10.7592/fejf2009.42.valk.'},
    {"badge": "sacred_sites", "ref_id": "R5", "source": 'A. Kaasik, "Sacred Natural Sites of Estonia," 2018. [Online]. Available: https://rpr.gov.lv/wp-content/uploads/2018/01/Sacred-Natural-Sites-of-Estonia.pdf (accessed Aug. 21, 2026).'},
    {"badge": "sacred_sites", "ref_id": "R6", "source": '"Maausk," Keskkonnaportaal (Estonian Environment Portal). [Online]. Available: https://keskkonnaportaal.ee/en/node/10161 (accessed Aug. 21, 2026).'},
    # Noble Trees
    {"badge": "noble_trees", "ref_id": "R7", "source": 'Society of Irish Foresters, "The Old Irish tree-list," Irish Forestry. [Online]. Available: https://journal.societyofirishforesters.ie/index.php/forestry/article/download/9882/8974 (accessed Aug. 21, 2026).'},
    {"badge": "noble_trees", "ref_id": "R8", "source": 'F. Kelly, "The Use of Ireland\'s Woodland in Medieval Times," School of Celtic Studies, Dublin Institute for Advanced Studies. [Online]. Available: https://www.woodlandsofireland.com/wp-content/uploads/The-Use-of-Irelands-Woodland-in-Medieval-times-by-Fergus-Kelly.pdf (accessed Aug. 21, 2026).'},
    {"badge": "noble_trees", "ref_id": "R9", "source": 'S. Barrett, "The Old Irish Tree List," LEIGHEAS Project Blog, Maynooth University. [Online]. Available: https://leigheas.maynoothuniversity.ie/?p=270 (accessed Aug. 21, 2026).'},
    {"badge": "noble_trees", "ref_id": "R10", "source": 'L. S. Joseph and B. Drayton, "Trees and Tradition in Early Ireland," Studia Celtica Fennica, vol. 17, pp. 55–, 2020–2021. [Online]. Available: https://journal.fi/scf/article/download/109499/68280/227908 (accessed Aug. 21, 2026).'},
    {"badge": "noble_trees", "ref_id": "R11", "source": '"Dairthech," Wikipedia. [Online]. Available: https://en.wikipedia.org/wiki/Dairthech (accessed Aug. 21, 2026).'},
    {"badge": "noble_trees", "ref_id": "R12", "source": 'L. O\'Brien, "Sacred Trees in Early Ireland." [Online]. Available: https://loraobrien.ie/?p=441 (accessed Aug. 21, 2026).'},
    # Nature's Pharmacy
    {"badge": "natures_pharmacy", "ref_id": "R13", "source": 'The Newt in Somerset, "Recipe: Soothing Lime Blossom Tea." [Online]. Available: https://thenewtinsomerset.com/journal/post/recipe-soothing-lime-blossom-tea (accessed Aug. 21, 2026).'},
    {"badge": "natures_pharmacy", "ref_id": "R14", "source": 'WebMD, "Health Benefits of Linden Tea." [Online]. Available: https://webmd.com/diet/health-benefits-linden-tea (accessed Aug. 21, 2026).'},
    {"badge": "natures_pharmacy", "ref_id": "R15", "source": 'H. Hemmes, "Hawthorn 1000 60 Capsules," National Pharmacies. [Online]. Available: https://www.nationalpharmacies.com.au/?p=153661 (accessed Aug. 21, 2026).'},
    {"badge": "natures_pharmacy", "ref_id": "R16", "source": '"Crataegus monogyna – Hawthorn," IMTL Plant Database. [Online]. Available: https://plant.imtl.gr/?p=456 (accessed Aug. 21, 2026).'},
    {"badge": "natures_pharmacy", "ref_id": "R17", "source": 'I. Svanberg et al., "Uses of tree saps in northern and eastern parts of Europe," Acta Societatis Botanicorum Poloniae, 2012. [Online]. Available: https://arca.unive.it/retrieve/e4239ddc-8319-7180-e053-3705fe0a3322/Treesaps_ASPB.pdf (accessed Aug. 21, 2026).'},
    {"badge": "natures_pharmacy", "ref_id": "R18", "source": 'N. Lagoni, "Esche in der Volksheilkunde und Pharmazie," Bayerische Landesanstalt für Wald und Forstwirtschaft. [Online]. Available: https://lfl.bayern.de/mam/cms04/wissenstransfer/dateien/w34_esche_in_der_volkheilkunde_und_pharmazie.pdf (accessed Aug. 21, 2026).'},
    {"badge": "natures_pharmacy", "ref_id": "R19", "source": '"Eschen: Berühmte Persönlichkeiten," wissen.de. [Online]. Available: https://www.wissen.de/bildwb/eschen-beruehmte-persoenlichkeiten (accessed Aug. 21, 2026).'},
    {"badge": "natures_pharmacy", "ref_id": "R29", "source": 'P. J. Herbal, "Corn silk (Zea mays) – traditional uses in Native American and Chinese medicine," Journal of Ethnopharmacology, 2010.'},
]


# ==================================================
# REFERENCES (plants)
# ==================================================

REFERENCES = [
    # --- Birke [1-3] ---
    {"plant": "birke_(betula_pendula)", "source": '"Die Birken: Vielseitige Begleiter der Menschen von einst bis heute," Bayern.de, 2023. https://www.lwf.bayern.de/waldbesitz-forstpolitik/waldfunktionen-landesplanung/342794/index.php (accessed Aug. 11, 2026).'},
    {"plant": "birke_(betula_pendula)", "source": 'I. Joost, „Hängebirke (Betula pendula)," NABU Dreisamtal, Baumlehrpfad in Kirchzarten. https://www.nabu-dreisamtal.de/infotafeln-im-dreisamtal/baumlehrpfad-in-kirchzarten/hängebirke/ (accessed Aug. 11, 2026).'},
    {"plant": "birke_(betula_pendula)", "source": 'A. Kaasik, "Conserving sacred natural sites in Estonia," in The Diversity of Sacred Lands in Europe: Proceedings of the Third Workshop of the Delos Initiative – Inari/Aanaar 2010, J.-M. Mallarach, T. Papayannis, and R. Väisänen, Eds. Gland, Switzerland: IUCN; Vantaa, Finland: Metsähallitus Natural Heritage Services, 2012, pp. 61–73.'},
    # --- Bohne [4-5] ---
    {"plant": "bohne_(faba)", "source": '"Rainmakers From the Gods - The Ceremonies - Powamuya: Bean Dance | Peabody Museum of Archaeology & Ethnology," Harvard.edu, 2025. https://peabody.harvard.edu/galleries/rainmakers-gods-ceremonies-powamuya-bean-dance (accessed Aug. 21, 2026).'},
    {"plant": "bohne_(faba)", "source": 'C. Huffman, "Pythagoras (Stanford Encyclopedia of Philosophy/Summer 2007 Edition)," Stanford.edu. https://plato.stanford.edu/archives/sum2007/entries/pythagoras/ (accessed Aug. 21, 2026).'},
    # --- Eibe [6-7] ---
    {"plant": "eibe_(taxus_baccata)", "source": 'National Museum of Ireland – Country Life, Of Fairies and Fairy Folk: Yew (Taxus baccata), Information Sheet, National Museum of Ireland, Ireland.'},
    {"plant": "eibe_(taxus_baccata)", "source": 'G.-A.-U. G. Öffentlichkeitsarbeit, "Mythen - Georg-August-Universität Göttingen," Uni-goettingen.de. https://www.uni-goettingen.de/de/mythen/60944.html (accessed Aug. 12, 2026).'},
    # --- Eiche [8] ---
    {"plant": "eiche_(quercus)", "source": 'G.-A.-U. G. Öffentlichkeitsarbeit, "Mythologie und Brauchtum - Georg-August-Universität Göttingen," Uni-goettingen.de. https://www.uni-goettingen.de/de/mythologie+und+brauchtum/16703.html (accessed Aug. 21, 2026).'},
    # --- Esche [9-10] ---
    {"plant": "esche_(fraxinus_excelsior)", "source": 'National Museum of Ireland – Country Life, Of Fairies and Fairy Folk: Ash (Fraxinus excelsior), Information Sheet, National Museum of Ireland, Ireland.'},
    {"plant": "esche_(fraxinus_excelsior)", "source": 'G.-A.-U. G. Öffentlichkeitsarbeit, "Mythologie und Volksheilkunde - Georg-August-Universität Göttingen," Uni-goettingen.de. https://www.uni-goettingen.de/de/mythologie+und+volksheilkunde/33096.html (accessed Aug. 14, 2026).'},
    # --- Gemeiner Schneeball [11] ---
    {"plant": "gemeiner_schneeball_(viburnum_opulus)", "source": 'Staatlicher Forstdienst der Ukraine: "Дерева - символи України," Forest.gov.ua, 2024. https://se.forest.gov.ua/press-sluzhba/novini-upravlinnya/dereva-simvoli-ukraini.html (accessed Aug. 12, 2026).'},
    # --- Gerste [12-13] ---
    {"plant": "gerste_(hordeum_vulgare)", "source": 'Y. Li, C. Long, K. Kato, C. Yang, and K. Sato, "Indigenous knowledge and traditional conservation of hulless barley (Hordeum vulgare) germplasm resources in the Tibetan communities of Shangri-la, Yunnan, SW China," Genetic Resources and Crop Evolution, vol. 58, no. 5, pp. 645–655, Oct. 2010, doi: 10.1007/s10722-010-9604-2.'},
    {"plant": "gerste_(hordeum_vulgare)", "source": 'C. S. Weckerle, F. K. Huber, Y. Yongping, and S. Weibang, "The Role of Barley among the Shuhi in the Tibetan Cultural Area of the Eastern Himalayas," Economic Botany, vol. 59, no. 4, pp. 386–390, Aug. 2005, doi: 10.1663/0013-0001(2005)059[0386:noep]2.0.co;2.'},
    # --- Hasel [14-15] ---
    {"plant": "hasel_(corylus_avellana)", "source": 'National Museum of Ireland – Country Life, Of Fairies and Fairy Folk: Hazel (Corylus avellana), Information Sheet, National Museum of Ireland, Ireland.'},
    {"plant": "hasel_(corylus_avellana)", "source": 'U. C. Cork, "Corylus avellana | University College Cork," University College Cork, 2024. https://www.ucc.ie/en/tree-explorers/trees/a-z/corylusavellana/ (accessed Aug. 12, 2026).'},
    # --- Kirschbluete [16-19] ---
    {"plant": "kirschbluete_(prunus_spp)", "source": 'B. F., "Cherry Trees in Japanese Folklore," Brooklyn Botanic Garden, Apr. 25, 2016. https://www.bbg.org/article/cherry_trees_in_japanese_folklore (accessed Aug. 15, 2026).'},
    {"plant": "kirschbluete_(prunus_spp)", "source": '"What Do Cherry Blossoms Represent in Japanese Culture? - JAPAN AIRLINES (JAL)," JAPAN AIRLINES (JAL) Official Site, 2023. https://www.jal.co.jp/in/en/guide-to-japan/experiences/cherry-blossom/what-do-cherry-blossoms-represent/index.html (accessed Aug. 15, 2026).'},
    {"plant": "kirschbluete_(prunus_spp)", "source": 'G.-A.-U. G. Öffentlichkeitsarbeit, "Mythologie - Georg-August-Universität Göttingen," Uni-goettingen.de. https://www.uni-goettingen.de/de/mythologie/46487.html (accessed Aug. 15, 2026).'},
    {"plant": "kirschbluete_(prunus_spp)", "source": 'G.-A.-U. G. Öffentlichkeitsarbeit, "Brauchtum - Georg-August-Universität Göttingen," Uni-goettingen.de. https://www.uni-goettingen.de/de/brauchtum/46488.html (accessed Aug. 15, 2026).'},
    # --- Kuerbis [20-22] ---
    {"plant": "kuerbis_(cucurbita_spp)", "source": '"Rainmakers From the Gods - The Ceremonies - Hakitonmuya: Plaza Dances, Footraces," Harvard.edu, 2024. https://peabody.harvard.edu/galleries/rainmakers-gods-ceremonies-hakitonmuya-plaza-dances-footraces (accessed Aug. 21, 2026).'},
    {"plant": "kuerbis_(cucurbita_spp)", "source": '"Patung," Wikipedia. July 18, 2025. Available: https://en.wikipedia.org/w/index.php?title=Patung&oldid=1301098126 (accessed Aug. 21, 2026).'},
    {"plant": "kuerbis_(cucurbita_spp)", "source": 'Wikipedia Contributors, "Jack-o\'-lantern," Wikipedia, Nov. 01, 2021. https://en.wikipedia.org/wiki/Jack-o%27-lantern (accessed Aug. 21, 2026).'},
    # --- Linde [23-25] ---
    {"plant": "linde_(tilla_cordata)", "source": 'G.-A.-U. G. Öffentlichkeitsarbeit, "Symbolik der Linde - Georg-August-Universität Göttingen," Uni-goettingen.de. https://www2.uni-goettingen.de/de/symbolik%2Bder%2Blinde/41770.html (accessed Aug. 13, 2026).'},
    {"plant": "linde_(tilla_cordata)", "source": 'G.-A.-U. G. Öffentlichkeitsarbeit, "Mythologie - Georg-August-Universität Göttingen," Uni-goettingen.de. https://www.uni-goettingen.de/de/mythologie/41688.html (accessed Aug. 13, 2026).'},
    {"plant": "linde_(tilla_cordata)", "source": 'A. Kaasik, "Conserving sacred natural sites in Estonia," in The Diversity of Sacred Lands in Europe: Proceedings of the Third Workshop of the Delos Initiative – Inari/Aanaar 2010, J.-M. Mallarach, T. Papayannis, and R. Väisänen, Eds. Gland, Switzerland: IUCN; Vantaa, Finland: Metsähallitus Natural Heritage Services, 2012, pp. 61–73.'},
    # --- Mais [26-30] ---
    {"plant": "mais_(zea_mays)", "source": '"myaamia miincipi: Growing Native Corn for the Smithsonian Folklife Festival," Smithsonian Folklife Festival, Aug. 25, 2025. https://festival.si.edu/blog/myaamia-miincipi-native-corn (accessed Aug. 14, 2026).'},
    {"plant": "mais_(zea_mays)", "source": '"Corn and Calendar Traditions | Living Maya Time," maya.nmai.si.edu. https://maya.nmai.si.edu/corn-and-maya-time/corn-and-calendar-traditions (accessed Aug. 14, 2026).'},
    {"plant": "mais_(zea_mays)", "source": 'Wikipedia Contributors, "Maya maize god," Wikipedia, Mar. 27, 2019. https://en.wikipedia.org/wiki/Maya_maize_god (accessed Aug. 14, 2026).'},
    {"plant": "mais_(zea_mays)", "source": 'E. Bullard, "Maya maize god | EBSCO," EBSCO Information Services, Inc., 2023. https://www.ebsco.com/research-starters/religion-and-philosophy/maya-maize-god (accessed Aug. 15, 2026).'},
    {"plant": "mais_(zea_mays)", "source": 'J. Dessein, E. Battaglini, and L. Horlings, Cultural Sustainability and Regional Development: Theories and Practices of Territorialisation. Routledge, 2017.'},
    # --- Waldkiefer [31-32] ---
    {"plant": "waldkierfer_(prinus_sylvestris)", "source": 'G.-A.-U. G. Öffentlichkeitsarbeit, "Symbol und Brauchtum - Georg-August-Universität Göttingen," Uni-goettingen.de. https://www.uni-goettingen.de/de/symbol+und+brauchtum/36747.html (accessed Aug. 12, 2026).'},
    {"plant": "waldkierfer_(prinus_sylvestris)", "source": 'H. Valk, "Sacred Natural Places of Estonia: Regional Aspects," Folklore: Electronic Journal of Folklore, vol. 42, pp. 45–66, 2009, doi: 10.7592/fejf2009.42.valk.'},
    # --- Weissdorn [33] ---
    {"plant": "weissdorn_(crataegus_monogyna)", "source": 'National Museum of Ireland – Country Life, Of Fairies and Fairy Folk: Hawthorn (Crataegus monogyna), Information Sheet, National Museum of Ireland, Ireland.'},
]

# Build plant → reference number map
PLANT_REFERENCE_NUMBERS = {}
for _ref_index, _ref_entry in enumerate(REFERENCES, start=1):
    PLANT_REFERENCE_NUMBERS.setdefault(_ref_entry["plant"], []).append(_ref_index)

# Build badge ref_id → absolute number map
BADGE_REF_ID_TO_NUM = {}
for _i, _br in enumerate(BADGE_REFERENCES, start=1):
    BADGE_REF_ID_TO_NUM[_br["ref_id"]] = len(REFERENCES) + _i


# ==================================================
# SESSION STATE
# ==================================================

st.session_state.setdefault("collection", [])
st.session_state.setdefault("page", "scan")
st.session_state.setdefault("show_plant_dialog", False)
st.session_state.setdefault("last_plant", None)
st.session_state.setdefault("last_confidence", None)
st.session_state.setdefault("show_badge_dialog", False)
st.session_state.setdefault("last_badge_key", None)

_query_page = st.query_params.get("page")
if _query_page:
    if _query_page in ("scan", "collection", "badges", "references"):
        st.session_state.page = _query_page
    st.query_params.clear()


# ==================================================
# NAVIGATION CSS
# ==================================================

active_page = st.session_state.page

navigation_css = """
<style>

.st-key-navigation {
    background: transparent !important;
    border: none !important;
    padding: 0;
    margin: 0 auto 30px auto;
    max-width: 720px;
}

.st-key-nav_collection_slot button,
.st-key-nav_scan_slot button,
.st-key-nav_badges_slot button {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #2e652d !important;
    font-family: 'Lora', serif !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    min-height: 130px;
    padding: 4px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    gap: 10px;
    transition: all 0.25s ease;
}

.st-key-nav_collection_slot button:hover,
.st-key-nav_scan_slot button:hover,
.st-key-nav_badges_slot button:hover {
    background: transparent !important;
    border: none !important;
    color: #234e22 !important;
}

.st-key-nav_collection_slot button::before,
.st-key-nav_scan_slot button::before,
.st-key-nav_badges_slot button::before {
    content: "";
    display: block;
    width: 62px;
    height: 62px;
    border-radius: 50%;
    background: #56224b;
    transition: all 0.25s ease;
}

@media (max-width: 600px) {
    .st-key-navigation { margin-bottom: 15px; }

    .st-key-nav_collection_slot button,
    .st-key-nav_scan_slot button,
    .st-key-nav_badges_slot button {
        min-height: 95px;
        font-size: 12px !important;
        gap: 5px;
    }

    .st-key-nav_collection_slot button::before,
    .st-key-nav_scan_slot button::before,
    .st-key-nav_badges_slot button::before {
        width: 45px;
        height: 45px;
    }
}

</style>
"""

active_nav_css_template = """
<style>

.st-key-nav_{page}_slot button {{
    color: #29482c !important;
    font-size: 20px !important;
    font-weight: 700 !important;
}}

.st-key-nav_{page}_slot button::before {{
    width: 92px !important;
    height: 92px !important;
    background: #5f8f63 !important;
}}

@media (max-width: 600px) {{
    .st-key-nav_{page}_slot button {{ font-size: 15px !important; }}

    .st-key-nav_{page}_slot button::before {{
        width: 65px !important;
        height: 65px !important;
    }}
}}

</style>
"""

if active_page in ("collection", "scan", "badges"):
    navigation_css += active_nav_css_template.format(page=active_page)

st.markdown(navigation_css, unsafe_allow_html=True)


# ==================================================
# HELPER: replace [Rn] tags with linked numbers
# ==================================================

def resolve_badge_refs(text):
    """Replace [R1], [R2], ... with linked reference numbers."""
    def _replace(match):
        rid = match.group(0)[1:-1]  # e.g. "R1"
        num = BADGE_REF_ID_TO_NUM.get(rid)
        if num:
            return (
                f'<a href="?page=references" '
                f'style="color:#56224b;text-decoration:none;font-weight:700;">'
                f'[{num}]</a>'
            )
        return match.group(0)
    return re.sub(r'$R\d+$', _replace, text)


# ==================================================
# PLANT INFORMATION POPUP
# ==================================================

@st.dialog("Plant Discovery", width="large", dismissible=False)
def show_plant_information():
    plant = st.session_state.last_plant
    confidence = st.session_state.last_confidence
    display_name = pretty_names.get(plant, plant)

    info_text = plant_information.get(
        plant, "More information about this plant will be added soon."
    )
    info_text = inspect.cleandoc(info_text).strip()

    info_html = (
        info_text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n\n", "</p><p>")
        .replace("\n", "<br>")
    )

    if os.path.exists(INFO_BACKGROUND_PATH):
        with open(INFO_BACKGROUND_PATH, "rb") as f:
            info_encoded = base64.b64encode(f.read()).decode()
        popup_background = (
            "background-image: linear-gradient("
            "rgba(255,255,255,0.82), rgba(255,255,255,0.82)), "
            f"url('data:image/jpeg;base64,{info_encoded}'); "
            "background-size: cover; background-position: center;"
        )
    else:
        popup_background = "background: rgba(245,248,243,0.97);"

    confidence_html = (
        "Previously discovered"
        if confidence is None
        else f"{confidence * 100:.1f}%"
    )

    ref_numbers = PLANT_REFERENCE_NUMBERS.get(plant, [])
    if ref_numbers:
        ref_links = ", ".join(
            f'<a href="?page=references" style="color:#56224b;text-decoration:none;font-weight:700;">[{n}]</a>'
            for n in ref_numbers
        )
        sources_html = f'<p style="margin:14px 0 0 0;font-size:13px;opacity:0.75;">Sources: {ref_links}</p>'
    else:
        sources_html = ""

    popup_html = f'''<div style="{popup_background} border-radius:26px; padding:28px; font-family:'Lora',serif; color:#2f3a1f;">
  <div class="popup-header">
    {get_popup_icon_html(plant)}
    <div class="popup-header-text">
      <div style="font-size:13px;opacity:0.65;letter-spacing:1px;margin-bottom:4px;">YOU DISCOVERED</div>
      <div style="font-family:'Playfair Display',serif;color:#56224b;font-size:32px;font-weight:700;margin-bottom:4px;">{display_name}</div>
      <div style="font-size:16px;opacity:0.80;">Confidence: <strong>{confidence_html}</strong></div>
    </div>
  </div>
  <div style="max-height:48vh;overflow-y:auto;background:rgba(255,255,255,0.76);border:1px solid rgba(100,120,100,0.20);border-radius:20px;padding:22px;font-size:18px;line-height:1.65;">
    <p style="margin:0;">{info_html}</p>
    {sources_html}
  </div>
</div>'''
    st.markdown(popup_html, unsafe_allow_html=True)

    left, middle, right = st.columns([2, 1, 2])
    with middle:
        if st.button("I See", use_container_width=True):
            st.session_state.show_plant_dialog = False
            st.rerun()


# ==================================================
# BADGE INFORMATION POPUP
# ==================================================

@st.dialog("New Achievement!", width="large", dismissible=False)
def show_badge_information():
    badge_key = st.session_state.last_badge_key
    badge = next((b for b in BADGES if b["key"] == badge_key), None)
    if badge is None:
        st.session_state.show_badge_dialog = False
        st.rerun()
        return

    badge_text = inspect.cleandoc(badge["text"]).strip()

    # 1. Replace [Rn] with linked numbers before escaping
    badge_text_linked = resolve_badge_refs(badge_text)

    # 2. Escape plain-text segments, preserve injected <a> tags
    link_pattern = re.compile(
        r'(<a href="\?page=references"[^>]*>.*?</a>)', re.DOTALL
    )
    parts = link_pattern.split(badge_text_linked)
    escaped_parts = []
    for part in parts:
        if link_pattern.match(part):
            escaped_parts.append(part)
        else:
            escaped_parts.append(
                part.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
            )
    badge_text_final = "".join(escaped_parts)

    # 3. Newlines → HTML paragraphs
    info_html = (
        badge_text_final
        .replace("\n\n", "</p><p>")
        .replace("\n", "<br>")
    )

    badge_refs = [r for r in BADGE_REFERENCES if r["badge"] == badge_key]
    if badge_refs:
        ref_links = ", ".join(
            f'<a href="?page=references" style="color:#56224b;text-decoration:none;font-weight:700;">'
            f'[{BADGE_REF_ID_TO_NUM[r["ref_id"]]}]</a>'
            for r in badge_refs
        )
        sources_html = f'<p style="margin:14px 0 0 0;font-size:13px;opacity:0.75;">Sources: {ref_links}</p>'
    else:
        sources_html = ""

    if os.path.exists(INFO_BACKGROUND_PATH):
        with open(INFO_BACKGROUND_PATH, "rb") as f:
            info_encoded = base64.b64encode(f.read()).decode()
        popup_background = (
            "background-image: linear-gradient("
            "rgba(255,255,255,0.82), rgba(255,255,255,0.82)), "
            f"url('data:image/jpeg;base64,{info_encoded}'); "
            "background-size: cover; background-position: center;"
        )
    else:
        popup_background = "background: rgba(245,248,243,0.97);"

    popup_html = f'''<div style="{popup_background} border-radius:26px; padding:28px; font-family:'Lora',serif; color:#2f3a1f;">
  <div class="popup-header">
    {get_badge_popup_icon_html(badge_key)}
    <div class="popup-header-text">
      <div style="font-size:13px;opacity:0.65;letter-spacing:1px;margin-bottom:4px;">NEW ACHIEVEMENT</div>
      <div style="font-family:'Playfair Display',serif;color:#56224b;font-size:32px;font-weight:700;margin-bottom:4px;">{badge["title"]}</div>
    </div>
  </div>
  <div style="max-height:48vh;overflow-y:auto;background:rgba(255,255,255,0.76);border:1px solid rgba(100,120,100,0.20);border-radius:20px;padding:22px;font-size:18px;line-height:1.65;">
    <p style="margin:0;">{info_html}</p>
    {sources_html}
  </div>
</div>'''
    st.markdown(popup_html, unsafe_allow_html=True)

    left, middle, right = st.columns([2, 1, 2])
    with middle:
        if st.button("I See", key="badge_dialog_close", use_container_width=True):
            st.session_state.show_badge_dialog = False
            st.rerun()


# ==================================================
# REFERENCES LINK
# ==================================================

st.markdown(
    '<div style="text-align:right; margin: 0 0 6px 0;">'
    '<a href="?page=references" style="font-family:\'Lora\',serif; font-size:14px; '
    'font-weight:500; color:#2f3a1f; text-decoration:none;">References</a>'
    '</div>',
    unsafe_allow_html=True,
)


# ==================================================
# NAVIGATION
# ==================================================

with st.container(key="navigation"):
    col1, col2, col3 = st.columns([1, 1, 1], gap="small")

    with col1, st.container(key="nav_collection_slot"):
        if st.button("Collection", key="nav_collection", use_container_width=True):
            st.session_state.page = "collection"
            st.rerun()

    with col2, st.container(key="nav_scan_slot"):
        if st.button("Scan", key="nav_scan", use_container_width=True):
            st.session_state.page = "scan"
            st.rerun()

    with col3, st.container(key="nav_badges_slot"):
        if st.button("Badges", key="nav_badges", use_container_width=True):
            st.session_state.page = "badges"
            st.rerun()


# ==================================================
# PAGE 1 — SCAN
# ==================================================

if st.session_state.page == "scan":
    st.title("Plant Scanner")

    st.markdown("""
<div class="main-card">
<h3>Let's discover your plant</h3>
<p>
Upload a photo of a plant and the model will try
to identify the species.
</p>
<p>
For the best result, use a clear photo with good
lighting and make sure the plant is easy to see.
</p>
</div>
""", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload plant photo", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        width, height = image.size
        min_side = 400
        if min(width, height) < min_side:
            st.warning(
                f"Your image is {width} × {height} px. "
                f"For better results, use an image where the shorter "
                f"side is at least {min_side} pixels."
            )

        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Identify Plant", use_container_width=True):
            with st.spinner("Analyzing plant..."):
                img_array = tf.keras.utils.img_to_array(image) 
                img_array = tf.image.resize(img_array, (256, 256))
                img_array = tf.expand_dims(img_array, axis=0)
                st.write("Input shape:", img_array.shape) 
                st.write("Input min:", float(tf.reduce_min(img_array))) 
                st.write("Input max:", float(tf.reduce_max(img_array))) 
                st.write("Input mean:", float(tf.reduce_mean(img_array)))

                st.write("Model parameters:", model.count_params())


                predictions = model.predict(img_array, verbose=0)
                st.write("Prediction vector:", predictions[0])
                st.write("Predicted index:", int(np.argmax(predictions[0])))
                
                predicted_index = int(np.argmax(predictions[0]))
                confidence = float(predictions[0][predicted_index])
                plant = class_names[predicted_index]

                st.session_state.last_plant = plant
                st.session_state.last_confidence = confidence
                st.session_state.show_plant_dialog = True

                if plant not in st.session_state.collection:
                    st.session_state.collection.append(plant)

                st.rerun()


# ==================================================
# PAGE 2 — COLLECTION
# ==================================================

elif st.session_state.page == "collection":
    st.title("My Collection")

    number_of_plants = len(st.session_state.collection)
    st.write(f"You have discovered **{number_of_plants} of {len(class_names)} plant types**.")
    st.progress(number_of_plants / len(class_names))

    if number_of_plants == 0:
        st.info("Your collection is still empty. Scan your first plant to get started.")
    else:
        collection_columns = st.columns(3, gap="medium")
        for index, plant in enumerate(st.session_state.collection):
            display_name = pretty_names.get(plant, plant)
            with collection_columns[index % 3]:
                tone = get_collection_card_tone(index)
                card_html = (
                    f'<div class="collection-card" style="background:{tone};">'
                    f'{get_plant_icon_html(plant)}'
                    f'<div class="collection-card-name">{display_name}</div>'
                    '</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)
                if st.button("Plant Info", key=f"collection_info_{index}", use_container_width=True):
                    st.session_state.last_plant = plant
                    st.session_state.last_confidence = None
                    st.session_state.show_plant_dialog = True
                    st.rerun()


# ==================================================
# PAGE 3 — BADGES
# ==================================================

elif st.session_state.page == "badges":
    st.title("Your Badges")
    st.write("Discover plants to unlock hidden achievements.")

    collected = st.session_state.collection
    badge_columns = st.columns(4, gap="medium")

    for col_idx, badge in enumerate(BADGES):
        unlocked = all(p in collected for p in badge["required_plants"])

        with badge_columns[col_idx % 4]:
            if unlocked:
                card_html = (
                    f'<div class="badge-card badge-unlocked">'
                    f'{get_badge_card_icon_html(badge["key"])}'
                    f'<div class="badge-title">{badge["title"]}</div>'
                    f'</div>'
                )
            else:
                card_html = '<div class="badge-card badge-locked"><div class="badge-dot">•</div></div>'

            st.markdown(card_html, unsafe_allow_html=True)

            if unlocked:
                if st.button(badge["title"], key=f"badge_btn_{badge['key']}", use_container_width=True):
                    st.session_state.last_badge_key = badge["key"]
                    st.session_state.show_badge_dialog = True
                    st.rerun()


# ==================================================
# PAGE 4 — REFERENCES
# ==================================================

elif st.session_state.page == "references":
    st.title("References")
    st.write("Sources used for the plant stories and badge texts in this app.")

    references_html = '<div class="main-card">'

    references_html += '<h3 style="margin-top:0;">Plant Stories</h3>'
    for index, entry in enumerate(REFERENCES, start=1):
        plant_label = pretty_names.get(entry["plant"], entry["plant"])
        references_html += (
            f'<p style="margin:0 0 14px 0;">'
            f'<strong style="color:#56224b;">[{index}]</strong> '
            f'<em>{plant_label}</em> — {entry["source"]}'
            f'</p>'
        )

    references_html += '<h3 style="margin-top:24px;">Badge Achievements</h3>'
    for entry in BADGE_REFERENCES:
        num = BADGE_REF_ID_TO_NUM[entry["ref_id"]]
        badge_label = next(
            (b["title"] for b in BADGES if b["key"] == entry["badge"]),
            entry["badge"]
        )
        references_html += (
            f'<p style="margin:0 0 14px 0;">'
            f'<strong style="color:#56224b;">[{num}]</strong> '
            f'<em>{badge_label}</em> — {entry["source"]}'
            f'</p>'
        )

    references_html += '</div>'
    st.markdown(references_html, unsafe_allow_html=True)


# ==================================================
# OPEN POPUPS
# ==================================================

if st.session_state.show_plant_dialog and st.session_state.last_plant is not None:
    show_plant_information()

if st.session_state.show_badge_dialog and st.session_state.last_badge_key is not None:
    show_badge_information()
