"""Feature engineering para el dataset de popularidad de Spotify, listo para usar en un Pipeline."""
import ast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer, KBinsDiscretizer, MultiLabelBinarizer, OneHotEncoder,
    PolynomialFeatures, StandardScaler, TargetEncoder,
)

# =============================================================================
# 0) Features de grupo: usan train + test juntos (NO usan popularity).
#    Van fuera del pipeline porque necesitan ver todas las filas a la vez.
# =============================================================================
ALBUM_PROFILE_SRC = ["danceability", "energy", "loudness", "speechiness", "acousticness",
                     "instrumentalness", "valence", "tempo", "duration_ms", "explicit"]


def add_group_features(train, test):
    """Añade a train y test (devuelve copias) features que cuentan o promedian sobre
    train + test juntos, sin usar popularity:
    album_size, release_size, conteos de artista, nº de álbumes y de tracks del primer
    artista, cuota del álbum dentro del artista y perfil de audio medio del álbum."""
    cols = ["track_id", "album_name", "artists"] + ALBUM_PROFILE_SRC
    both = pd.concat([train[cols], test[cols]], ignore_index=True)
    both["explicit"] = both["explicit"].astype(int)
    both_first = both["artists"].str.split(";").str[0]
    both_album_key = both["album_name"] + "||" + both_first

    first_n_albums = both.groupby(both_first)["album_name"].nunique()
    first_n_tracks = both_first.value_counts()
    album_key_counts = both_album_key.value_counts()
    album_profile = both.groupby(both_album_key)[ALBUM_PROFILE_SRC].mean().add_prefix("alb_mean_")
    artist_profile = both.groupby(both_first)[ALBUM_PROFILE_SRC].mean()

    album_counts = both["album_name"].value_counts()
    artists_str_counts = both["artists"].value_counts()                    # cadena completa "A;B"
    artist_counts = both["artists"].str.split(";").explode().value_counts()  # cada artista por separado

    pairs = (both.assign(artist=both["artists"].str.split(";"))
                 .explode("artist")[["track_id", "album_name", "artist"]])
    links = pairs.merge(pairs, on=["album_name", "artist"], suffixes=("", "_other"))
    n_related = links.groupby("track_id")["track_id_other"].nunique()

    out = []
    for df in (train, test):
        df = df.copy()
        df["album_size"] = df["album_name"].map(album_counts)     # tracks con ese album_name
        df["release_size"] = df["track_id"].map(n_related)        # ... y con algún artista en común

        per_artist = df["artists"].str.split(";").explode().map(artist_counts)
        df["artist_count_max"] = per_artist.groupby(level=0).max()   # tracks del artista más prolífico
        df["artist_count_sum"] = per_artist.groupby(level=0).sum()   # suma sobre todos sus artistas
        df["artists_str_count"] = df["artists"].map(artists_str_counts)

        first = df["artists"].str.split(";").str[0]
        album_key = df["album_name"] + "||" + first
        df["first_artist_n_albums"] = first.map(first_n_albums)          # álbumes distintos del artista
        df["first_artist_n_tracks"] = first.map(first_n_tracks)          # tracks del artista
        df["album_share_of_artist"] = album_key.map(album_key_counts) / df["first_artist_n_tracks"]
        df[album_profile.columns] = album_profile.reindex(album_key).to_numpy()  # audio medio del álbum
        # audio del track menos el audio medio de su (primer) artista
        own = df[ALBUM_PROFILE_SRC].astype(float).to_numpy()
        df[[f"dev_artist_{c}" for c in ALBUM_PROFILE_SRC]] = own - artist_profile.reindex(first).to_numpy()
        out.append(df)
    return out[0], out[1]


# =============================================================================
# 1) Features fila a fila: no aprenden nada de los datos (sin riesgo de fuga).
# =============================================================================
TRACK_PATTERNS = {
    "is_feat":         r"feat\.|\bft\.|\(with ",
    "is_live":         r"\blive\b|en vivo|ao vivo",
    "is_remix":        r"remix",
    "is_remaster":     r"remaster",
    "is_acoustic":     r"acoustic|acústic",
    "is_instrumental_name": r"instrumental",
    "is_edit":         r"\bedit\b",
    "has_brackets":    r"[\(\[]",
}

ALBUM_PATTERNS = {
    "alb_deluxe":      r"deluxe",
    "alb_soundtrack":  r"soundtrack|\bost\b|motion picture|from the",
    "alb_remaster":    r"remaster",
    "alb_edition":     r"edition|edición",
    "alb_live":        r"\blive\b|en vivo|ao vivo",
    "alb_compilation": r"greatest|best of|\bhits\b|collection|essentials|\bvol\b|vol\.|volume",
    "alb_generic":     r"\btop\b|playlist|\bmix\b|songs for|music for",
    "alb_christmas":   r"christmas|navidad|xmas",
    "alb_year":        r"\b(?:19|20)\d\d\b",
}


def _normalize(s):
    return (s.str.lower()
             .str.replace(r"\s*[\(\[].*?[\)\]]", "", regex=True)
             .str.replace(r"\s+-\s+.*$", "", regex=True)
             .str.strip())


def row_features(df):
    """Crea todas las features que solo dependen de la propia fila."""
    df = df.copy()

    # Géneros: texto "['a', 'b']" -> lista
    if isinstance(df["track_genre"].iloc[0], str):
        df["track_genre"] = df["track_genre"].apply(ast.literal_eval)
    df["n_genres"] = df["track_genre"].str.len()
    df["genre_combo"] = df["track_genre"].apply(lambda g: "|".join(sorted(g)))     # clave para TE

    # Artistas
    df["first_artist"] = df["artists"].str.split(";").str[0]                         # clave para stats
    artist_set = df["artists"].str.split(";").apply(lambda a: "|".join(sorted(a)))
    df["album_artist_set"] = df["album_name"] + "||" + artist_set                    # clave para TE
    df["artists_n"] = df["artists"].str.count(";") + 1
    df["artists_multiple"] = (df["artists_n"] > 1).astype(int)

    # Nombre del track y del álbum
    for col, pat in TRACK_PATTERNS.items():
        df[col] = df["track_name"].str.contains(pat, case=False, regex=True).astype(int)
    for col, pat in ALBUM_PATTERNS.items():
        df[col] = df["album_name"].str.contains(pat, case=False, regex=True).astype(int)
    df["name_len"] = df["track_name"].str.len()
    df["name_words"] = df["track_name"].str.split().str.len()
    df["album_name_len"] = df["album_name"].str.len()

    # Single vs álbum (requiere release_size de add_group_features)
    exact = df["track_name"] == df["album_name"]
    loose = _normalize(df["track_name"]) == _normalize(df["album_name"])
    alone = df["release_size"] == 1
    df["single_confidence"] = np.select(
        [exact & alone, loose & alone, loose & ~alone], [3, 2, 1], default=0)

    # Claves combinadas para target encoding y relación artista-álbum
    # Ojo: track_genre viene en orden alfabético, así que el "primer género" no es el principal.
    # Por eso las claves usan la combinación completa de géneros (genre_combo).
    df["artist_release_type"] = df["first_artist"] + "||" + np.where(df["single_confidence"] >= 2, "single", "album")
    df["artist_genre"] = df["first_artist"] + "||" + df["genre_combo"]
    df["song_artist"] = _normalize(df["track_name"]) + "||" + df["first_artist"]   # otras versiones de la canción
    df["album_is_self_titled"] = [int(a.lower() in b.lower())
                                  for a, b in zip(df["first_artist"], df["album_name"])]

    # Audio
    df["explicit"] = df["explicit"].astype(int)
    df["log_duration"] = np.log(df["duration_ms"])
    df["is_instrumental"] = (df["instrumentalness"] > 0.5).astype(int)
    df["is_live_audio"] = (df["liveness"] > 0.8).astype(int)
    df["speech_mixed"] = df["speechiness"].between(0.33, 0.66, inclusive="right").astype(int)
    df["speech_only"] = (df["speechiness"] > 0.66).astype(int)
    df["is_4_4"] = (df["time_signature"] == 4).astype(int)
    return df


# =============================================================================
# 2) Transformador para la lista de géneros (MultiLabelBinarizer no encaja
#    directamente en ColumnTransformer).
# =============================================================================
class GenreBinarizer(BaseEstimator, TransformerMixin):
    """Multi-hot de track_genre: una columna 0/1 por género (aprendidos en fit).

    drop_duplicates=True elimina los géneros cuya columna es idéntica a la de otro
    género en los datos de fit (p. ej. 'songwriter' = 'singer-songwriter' en este
    dataset). Necesario para LinearRegression; indiferente para los árboles.
    """

    def __init__(self, drop_duplicates=False):
        self.drop_duplicates = drop_duplicates

    def fit(self, X, y=None):
        lists = self._lists(X)
        self.mlb_ = MultiLabelBinarizer().fit(lists)
        keep = np.ones(len(self.mlb_.classes_), dtype=bool)
        if self.drop_duplicates:
            M = self.mlb_.transform(lists)
            _, first_idx = np.unique(M, axis=1, return_index=True)
            keep[:] = False
            keep[np.sort(first_idx)] = True
        self.keep_ = keep
        return self

    def transform(self, X):
        return self.mlb_.transform(self._lists(X))[:, self.keep_]

    def get_feature_names_out(self, input_features=None):
        return np.array([f"genre_{g}" for g in self.mlb_.classes_[self.keep_]], dtype=object)

    @staticmethod
    def _lists(X):
        col = X.iloc[:, 0] if hasattr(X, "iloc") else pd.Series(np.asarray(X)[:, 0])
        return col.tolist()


class ArtistTargetEncoder(BaseEstimator, TransformerMixin):
    """Target encoding por artista individual: la columna 'artists' se separa por ';'.

    Para cada track devuelve 5 columnas: media, máximo y mínimo de la popularidad media
    (suavizada) de sus artistas, cuántos de sus artistas son conocidos y cuántos tracks de
    entrenamiento suman. Un artista desconocido toma la media global.

    Igual que TargetEncoder de scikit-learn, fit_transform hace cross-fitting: la codificación
    de cada fila se calcula sin su propia popularidad (sin fuga). transform usa todos los
    datos de fit.
    """

    def __init__(self, smoothing=5, cv=5, random_state=42):
        self.smoothing = smoothing
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, y):
        self.stats_, self.global_mean_ = self._stats(self._col(X), y)
        return self

    def transform(self, X):
        return self._encode(self._col(X), self.stats_, self.global_mean_)

    def fit_transform(self, X, y):
        artists = self._col(X).reset_index(drop=True)
        y = np.asarray(y, dtype=float)
        out = np.zeros((len(artists), 5))
        folds = KFold(self.cv, shuffle=True, random_state=self.random_state)
        for fit_idx, enc_idx in folds.split(artists):
            stats, gm = self._stats(artists.iloc[fit_idx], y[fit_idx])
            out[enc_idx] = self._encode(artists.iloc[enc_idx], stats, gm)
        self.fit(X, y)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(["art_te_mean", "art_te_max", "art_te_min", "art_known", "art_train_tracks"],
                        dtype=object)

    @staticmethod
    def _col(X):
        col = X.iloc[:, 0] if hasattr(X, "iloc") else pd.Series(np.asarray(X)[:, 0])
        return col.astype(str)

    @staticmethod
    def _stats(artists, y):
        e = pd.DataFrame({"a": artists.str.split(";").to_numpy(), "y": np.asarray(y, dtype=float)})
        e = e.explode("a")
        return e.groupby("a")["y"].agg(["sum", "count"]), float(np.mean(y))

    def _encode(self, artists, stats, gm):
        m = self.smoothing
        enc = (stats["sum"] + m * gm) / (stats["count"] + m)
        e = artists.reset_index(drop=True).str.split(";").explode()
        v = e.map(enc)
        g = pd.DataFrame({"v": v, "c": e.map(stats["count"]).fillna(0), "known": v.notna()}).groupby(level=0)
        out = pd.DataFrame({
            "mean": g["v"].mean(), "max": g["v"].max(), "min": g["v"].min(),
            "known": g["known"].sum(), "tracks": g["c"].sum(),
        })
        out[["mean", "max", "min"]] = out[["mean", "max", "min"]].fillna(gm)
        return out.to_numpy(dtype=float)


class GroupStatsEncoder(BaseEstimator, TransformerMixin):
    """Estadísticas de la popularidad por grupo (p. ej. por artista): desviación, máximo,
    mínimo y mediana de los tracks de entrenamiento del grupo.

    sep=None: la columna es una clave simple (p. ej. first_artist).
    sep=";": la columna tiene varios valores por fila (p. ej. artists); se calculan las
    estadísticas de cada artista y se agregan por fila (máximo; la desviación, como media).

    Grupo desconocido o con un solo track (sin desviación) -> NaN, que los modelos de árboles
    usados aquí (sklearn >= 1.4, XGBoost, LightGBM, CatBoost) aceptan.
    fit_transform hace cross-fitting (sin fuga), igual que TargetEncoder.
    """

    STATS = ("std", "max", "min", "median")

    def __init__(self, sep=None, cv=5, random_state=42):
        self.sep = sep
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, y):
        self.stats_ = self._stats(self._col(X), np.asarray(y, dtype=float))
        return self

    def transform(self, X):
        return self._encode(self._col(X), self.stats_)

    def fit_transform(self, X, y):
        keys = self._col(X).reset_index(drop=True)
        y = np.asarray(y, dtype=float)
        out = np.full((len(keys), len(self.STATS)), np.nan)
        folds = KFold(self.cv, shuffle=True, random_state=self.random_state)
        for fit_idx, enc_idx in folds.split(keys):
            out[enc_idx] = self._encode(keys.iloc[enc_idx], self._stats(keys.iloc[fit_idx], y[fit_idx]))
        self.fit(X, y)
        return out

    def get_feature_names_out(self, input_features=None):
        prefix = input_features[0] if input_features is not None else "group"
        return np.array([f"{prefix}_pop_{s}" for s in self.STATS], dtype=object)

    @staticmethod
    def _col(X):
        col = X.iloc[:, 0] if hasattr(X, "iloc") else pd.Series(np.asarray(X)[:, 0])
        return col.astype(str)

    def _split(self, keys):
        k = keys.reset_index(drop=True)
        return k.str.split(self.sep).explode() if self.sep else k

    def _stats(self, keys, y):
        k = self._split(keys)
        return pd.DataFrame({"k": k.to_numpy(), "y": y[k.index]}).groupby("k")["y"].agg(list(self.STATS))

    def _encode(self, keys, stats):
        k = self._split(keys)
        vals = stats.reindex(k.to_numpy()).set_axis(k.index)
        if self.sep:
            g = vals.groupby(level=0)
            vals = pd.concat([g["std"].mean(), g["max"].max(), g["min"].max(), g["median"].max()], axis=1)
            vals = vals.reindex(range(len(keys)))
        return vals.to_numpy(dtype=float)


class RelativeTargetEncoder(BaseEstimator, TransformerMixin):
    """Diferencia entre el target encoding de dos columnas: TE(columna 1) - TE(columna 2).
    Con [first_artist, genre_combo] mide cuánto más (o menos) popular es el artista que la
    media de su combinación de géneros. Usa TargetEncoder de scikit-learn, así que fit_transform hace
    cross-fitting (sin fuga)."""

    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, y):
        self.te_ = _target_encoder(self.random_state).fit(X, y)
        return self

    def transform(self, X):
        enc = self.te_.transform(X)
        return (enc[:, 0] - enc[:, 1]).reshape(-1, 1)

    def fit_transform(self, X, y):
        self.te_ = _target_encoder(self.random_state)
        enc = self.te_.fit_transform(X, y)
        return (enc[:, 0] - enc[:, 1]).reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        a, b = input_features if input_features is not None else ("col1", "col2")
        return np.array([f"te_{a}_minus_{b}"], dtype=object)


class TextRidgeEncoder(BaseEstimator, TransformerMixin):
    """Modelo de texto apilado (stacking): TF-IDF de palabras (1-2) y de fragmentos de 2-4
    caracteres de cada columna de texto -> Ridge -> una columna con la popularidad predicha
    solo a partir del texto. Generaliza a artistas y álbumes nuevos (palabras como
    'Orchestra', 'Cast', 'White Noise', idioma, alfabeto...).
    fit_transform hace cross-fitting (sin fuga): cada fila se predice con un Ridge que no la vio.
    """

    def __init__(self, alpha=3.0, cv=5, random_state=42):
        self.alpha = alpha
        self.cv = cv
        self.random_state = random_state

    def _vectorizers(self, n_cols):
        from sklearn.feature_extraction.text import TfidfVectorizer
        return [(TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                 TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, sublinear_tf=True,
                                 max_features=100_000))
                for _ in range(n_cols)]

    @staticmethod
    def _texts(X):
        df = X if hasattr(X, "iloc") else pd.DataFrame(np.asarray(X))
        return [df.iloc[:, i].astype(str).str.replace(";", " ; ").reset_index(drop=True)
                for i in range(df.shape[1])]

    def _fit_models(self, texts, y):
        from scipy.sparse import hstack
        from sklearn.linear_model import Ridge
        vecs = self._vectorizers(len(texts))
        mats = []
        for (w, c), t in zip(vecs, texts):
            mats += [w.fit_transform(t), c.fit_transform(t)]
        return vecs, Ridge(alpha=self.alpha).fit(hstack(mats).tocsr(), y)

    @staticmethod
    def _predict(vecs, ridge, texts):
        from scipy.sparse import hstack
        mats = []
        for (w, c), t in zip(vecs, texts):
            mats += [w.transform(t), c.transform(t)]
        return ridge.predict(hstack(mats).tocsr())

    def fit(self, X, y):
        self.vecs_, self.ridge_ = self._fit_models(self._texts(X), np.asarray(y, dtype=float))
        return self

    def transform(self, X):
        return self._predict(self.vecs_, self.ridge_, self._texts(X)).reshape(-1, 1)

    def fit_transform(self, X, y):
        texts, y = self._texts(X), np.asarray(y, dtype=float)
        out = np.zeros(len(y))
        for fit_idx, enc_idx in KFold(self.cv, shuffle=True, random_state=self.random_state).split(texts[0]):
            vecs, ridge = self._fit_models([t.iloc[fit_idx] for t in texts], y[fit_idx])
            out[enc_idx] = self._predict(vecs, ridge, [t.iloc[enc_idx] for t in texts])
        self.fit(X, y)
        return out.reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array(["text_ridge_pred"], dtype=object)


# =============================================================================
# 3) Listas de columnas
# =============================================================================
FLAG_COLS = (list(TRACK_PATTERNS) + list(ALBUM_PATTERNS) + [
    "explicit", "mode", "artists_multiple", "is_instrumental", "is_live_audio",
    "speech_mixed", "speech_only", "is_4_4",
])
COUNT_COLS = ["artists_n", "n_genres", "name_len", "name_words", "album_name_len",
              "album_size", "release_size", "single_confidence"]
AUDIO_COLS = ["danceability", "energy", "loudness", "speechiness", "acousticness",
              "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
              "key", "time_signature"]

QUAD_COLS = ["log_duration", "loudness", "valence"]          # U invertida -> x y x²
DECILE_COLS = ["speechiness", "acousticness"]                # deciles
# n_genres se excluye del modelo lineal: es exactamente la suma de las columnas de género
LINEAR_SCALED = (["danceability", "energy", "instrumentalness", "liveness", "tempo"]
                 + [c for c in COUNT_COLS if c != "n_genres"])


# Preprocesador de árboles (versión actual): columnas originales + features del notebook
# para los textos + target encoding de los textos.
ORIGINAL_NUM_COLS = ["duration_ms", "explicit", "danceability", "energy", "key", "loudness",
                     "mode", "speechiness", "acousticness", "instrumentalness", "liveness",
                     "valence", "tempo", "time_signature"]
TEXT_FEATURE_COLS = (["artists_n", "artists_multiple"] + list(TRACK_PATTERNS)
                     + ["name_len", "name_words", "release_size", "single_confidence"])
TEXT_TE_COLS = ["artists", "track_name", "album_name"]
ARTIST_COUNT_COLS = ["artist_count_max", "artist_count_sum", "artists_str_count"]   # de add_group_features
EXTRA_TE_COLS = ["album_artist_set", "genre_combo",                                 # claves de row_features
                 "artist_release_type", "artist_genre", "song_artist"]
ALBUM_EXTRA_COLS = (list(ALBUM_PATTERNS) + ["album_size", "album_name_len",
                    "first_artist_n_albums", "first_artist_n_tracks", "album_share_of_artist"]
                    + [f"alb_mean_{c}" for c in ALBUM_PROFILE_SRC]
                    + [f"dev_artist_{c}" for c in ALBUM_PROFILE_SRC]
                    + ["album_is_self_titled"])


def _target_encoder(random_state=42):
    return TargetEncoder(target_type="continuous", smooth="auto",
                         cv=KFold(5, shuffle=True, random_state=random_state))


# =============================================================================
# 4) Dos preprocesadores: uno para modelos lineales y otro para árboles
# =============================================================================
def make_linear_preprocessor(random_state=42):
    """Sin columnas redundantes (matriz de rango completo), apto para LinearRegression:
    los one-hot eliminan la primera categoría y los géneros duplicados se quitan."""
    quad = [(f"{c}_quad", make_pipeline(StandardScaler(),
                                        PolynomialFeatures(2, include_bias=False)), [c])
            for c in QUAD_COLS]
    deciles = make_pipeline(
        KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile",
                         quantile_method="averaged_inverted_cdf"),
        OneHotEncoder(drop="first", sparse_output=False),       # decil 1 = referencia
    )
    return ColumnTransformer(
        quad + [
            ("deciles", deciles, DECILE_COLS),
            ("key", OneHotEncoder(drop="first", sparse_output=False), ["key"]),  # Do = referencia
            ("genre", GenreBinarizer(drop_duplicates=True), ["track_genre"]),
            ("album_te", make_pipeline(_target_encoder(random_state), StandardScaler()), ["album_name"]),
            ("scaled", StandardScaler(), LINEAR_SCALED),
            ("flags", "passthrough", FLAG_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_tree_preprocessor(random_state=42):
    """Árboles: 14 columnas numéricas originales tal cual + géneros multi-hot + las features
    de texto del notebook (artists, track_name, single_confidence) + target encoding de
    artists, track_name y album_name + target encoding por artista individual + conteos de
    artista + target encoding de (álbum, conjunto de artistas) y de la combinación de géneros
    + estadísticas de popularidad del artista (primer artista y cada artista individual)
    + palabras clave, tamaño y perfil de audio del álbum.
    Todo lo que usa popularity hace cross-fitting dentro del pipeline."""
    return ColumnTransformer(
        [
            ("genre", GenreBinarizer(), ["track_genre"]),
            ("text_te", _target_encoder(random_state), TEXT_TE_COLS + EXTRA_TE_COLS),
            ("artist_te", ArtistTargetEncoder(random_state=random_state), ["artists"]),
            ("first_artist_stats", GroupStatsEncoder(random_state=random_state), ["first_artist"]),
            ("artists_stats", GroupStatsEncoder(sep=";", random_state=random_state), ["artists"]),
            ("artist_vs_genre", RelativeTargetEncoder(random_state), ["first_artist", "genre_combo"]),
            ("genre_stats", GroupStatsEncoder(sep="|", random_state=random_state), ["genre_combo"]),
            ("text_model", TextRidgeEncoder(random_state=random_state), TEXT_TE_COLS),
            ("raw", "passthrough",
             ORIGINAL_NUM_COLS + TEXT_FEATURE_COLS + ARTIST_COUNT_COLS + ALBUM_EXTRA_COLS + ["n_genres"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_tree_preprocessor_full(random_state=42):
    """Versión anterior (todo el feature engineering, target encoding solo de album_name).
    Se conserva para comparar; en la prueba rápida quedó peor que make_tree_preprocessor."""
    return ColumnTransformer(
        [
            ("genre", GenreBinarizer(), ["track_genre"]),
            ("album_te", _target_encoder(random_state), ["album_name"]),
            ("raw", "passthrough", AUDIO_COLS + COUNT_COLS + FLAG_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


ROW_FEATURES = FunctionTransformer(row_features)
