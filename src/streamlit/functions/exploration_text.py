import streamlit as st 
from itertools import cycle 
import altair as alt
import pickle 
import pandas as pd
import matplotlib.pyplot as plt 
from wordcloud import WordCloud
#from functions.tools_st import convert_to_str, filepath_explo
import numpy as np
import src.tools.tools as tools
from pathlib import Path

from src.streamlit.functions.tools_st import show_zoomable_image, convert_to_str

def explo_summary(X, df_cat, df_stats):
    st.header("Structure des données")
    st.write("La structure des données fournies par le challenge ENS Data est décrite dans la figure ci-dessous :")
    img_path = "data/dataviz/schema_data.png"
    show_zoomable_image(img_path=img_path, caption="Schéma de la structure des données ")

    st.write("L'objectif du challenge est fournir un fichier y_pred contenant les prédictions des produits référencés dans le fichier *X_test*.")
    st.header("Déséquilibre du jeu de données")
    st.write("La figure suivante montre le nombre de codes types de produits dans le jeu de données d'entraînement :")

    chart = alt.Chart(df_stats).mark_bar().encode(
        x=alt.X('prdtypecode:N', sort=list(df_stats['prdtypecode'].values)),
        y='count:Q',
        tooltip = ['prdtypecode', 'Catégorie', 'count', 'pourcentage'],
        color=alt.Color('count', scale=alt.Scale(scheme='viridis', reverse=True))  # Apply Viridis color scale
        ).properties(width=600, height=400)
    st.altair_chart(chart, use_container_width=True)

    st.header("Identification manuelle des codes types de produits")
    st.write("Le challenge Rakuten ne fournit pas de définition précise des codes types de produits. Afin d'avoir une idée du type de produit associé à chaque code type, une exploration manuelle des données a été réalisée afin d'essayer de définir une catégorie de produits. Toutefois, cette identification manuelle peut s'avérer imprécise étant donnée la volumétrie importante du jeu de données.")
    
    df_cat.index = df_cat.index.astype(str)
    st.dataframe(df_cat)

    df_cat['Title'] = df_cat.index + " - " + df_cat['Catégorie']
    option_titles = list(df_cat['Title'].values)
    selected_title = st.selectbox('Choisir une catégorie de produits pour afficher 5 produits tirés aléatoirement de cette catégorie : ', option_titles)
    df_cat.index = df_cat.index.astype(int)
    selected_prdtypecode = df_cat.loc[df_cat['Title']==selected_title].index[0]
    df_select = X.loc[X['prdtypecode']==selected_prdtypecode].sample(5)
    df_select = df_select.sort_index()
    df_select['Filepath'] = df_select.apply(lambda row : tools.get_filepath_train(row['productid'], row['imageid']), axis=1)

    images = df_select['Filepath'].values
    captions = df_select['designation'].values

    cols = cycle(st.columns(5))
    for img, caption in zip(images, captions):
        next(cols).image(img, width=150, caption=caption)


def explo_textdata(X, x_train, y_train, x_test, df_cat):
    st.header("Données manquantes")
    st.write("Seule la description dans les jeux de données X_train et X_test contiennent des données manquantes. Les autres données sont systématiquement renseignées. Les jeux de données X_train et X_test contiennent chacune environ 35% de données manquantes pour la description des produits.")


    train_nan = x_train.isna().sum()
    test_nan = x_test.isna().sum()
    df_nan = pd.concat([train_nan, test_nan], axis=1)
    df_nan = df_nan.rename(columns={0: "X_train", 1: "X_test"})
    df_nan['feature'] = df_nan.index
    df_nan = df_nan.melt("feature", var_name='DataFrame', value_name="Nombre")
    df_nan['pourcentage'] = 0.
    df_nan.loc[df_nan['DataFrame']=='X_train', 'pourcentage'] = df_nan.loc[df_nan['DataFrame']=='X_train', 'Nombre'] / x_train.shape[0] * 100.
    df_nan.loc[df_nan['DataFrame']=='X_test', 'pourcentage'] = df_nan.loc[df_nan['DataFrame']=='X_test', 'Nombre'] / x_test.shape[0] * 100.
    df_nan['pourcentage'] = df_nan['pourcentage'].apply(convert_to_str)

    chart_nan = alt.Chart(df_nan).mark_bar().encode(
        x=alt.X('feature', sort=list(train_nan.index)),
        y="Nombre",
        color='DataFrame',
        tooltip=['DataFrame', 'feature', 'Nombre', 'pourcentage']
    ).properties(width=200, height=400, title="Nombre de NaN pour chaque feature des jeux de données").configure_title(anchor='middle')
    st.altair_chart(chart_nan)

    st.write("Les catégories du produit sont inégalement impactés par le nombre de descriptions manquantes. Le graphique suivant montre le nombre de descriptions manquantes par code type de produit :")
    df_train = pd.merge(left=x_train.isna(), right=y_train, left_index=True, right_index=True, how='left')
    stats_nans = df_train.groupby(by='prdtypecode')['description'].agg(['sum', 'count'])
    stats_nans['Ratio NaN'] = stats_nans['sum'] / stats_nans['count']
    stats_nans['Ratio PrdTypeCode'] = stats_nans['count'] / X.shape[0]
    stats_nans['prdtypecode'] = stats_nans.index
    stats_nans = pd.merge(left=stats_nans, right=df_cat, how='left', left_index=True, right_index=True)
    stats_nans['prdtypecode'] = stats_nans['prdtypecode'].astype(str)
    chart = alt.Chart(stats_nans).mark_circle().encode(
        x='Ratio PrdTypeCode',
        y='Ratio NaN',
        color=alt.Color('Ratio NaN', legend=None, scale=alt.Scale(scheme='blues', reverse=True)),
        size=alt.Size('Ratio NaN', scale=alt.Scale(range=[100, 350])),
        tooltip=['prdtypecode', 'Catégorie', 'Ratio PrdTypeCode', 'Ratio NaN']
    ).properties(width=600, height=500, title="Ratio des NaN en fonction du ratio Code Type dans X_train").configure_title(anchor='middle').interactive()
    st.altair_chart(chart)
    st.write("Certaines catégories (par exemple : 1180 et 2462) de produits ont un nombre important de descriptions manquantes alors qu'elles représentent chacune moins de 2% du volume de données")

    st.header("Longueur des données textuelles")
    st.write("La distribution des longueurs cumulées de la désignation et de la description de chaque produit permet de constater qu'une partie des produits ont peu de données textuelles renseignées.")
    stat_text_train = y_train
    stat_text_train['Données textuelles'] = x_train['designation'] + x_train.fillna("")['description']
    stat_text_train['Longueur données textuelles'] = stat_text_train['Données textuelles'].apply(len)
    stat_text_train.loc[stat_text_train['Longueur données textuelles']>4000, 'Longueur données textuelles'] = 4000

    hist_len_train = alt.Chart(stat_text_train).mark_bar().encode(
        alt.X('Longueur données textuelles:Q', bin=alt.Bin(maxbins=100), title="Longueur Données Textuelles"),
        y=alt.Y('count()', title="Nb de produits",
        axis=alt.Axis())#titleColor="#1f77b4"))
    )
    kde_len_train = alt.Chart(stat_text_train).transform_density('Longueur données textuelles', 
                                                as_=['Longueur données textuelles', 'densité']).mark_line(color='#29A989').encode(
                                                    x='Longueur données textuelles:Q',
                                                    y=alt.Y('densité:Q', axis=alt.Axis(titleColor='#29A989', title='Densité'))
                                                )
    mixed_char_train = alt.layer(hist_len_train, kde_len_train).resolve_scale(
        y='independent'
    ).properties(width=600, height=400, title="Distribution des longueurs de données textuelles dans X_train").configure_title(anchor='middle')

    st.altair_chart(mixed_char_train)

    stat_text_test = x_test
    stat_text_test['Données textuelles'] = stat_text_test['designation'] + stat_text_test.fillna("")['description']
    stat_text_test['Longueur données textuelles'] = stat_text_test['Données textuelles'].apply(len)
    stat_text_test.loc[stat_text_test['Longueur données textuelles']>4000, 'Longueur données textuelles'] = 4000

    hist_len_test = alt.Chart(stat_text_test).mark_bar().encode(
        alt.X('Longueur données textuelles:Q', bin=alt.Bin(maxbins=100), title="Longueur Données Textuelles"),
        y=alt.Y('count()', title="Nb de produits",
        axis=alt.Axis())#titleColor="#1f77b4"))
    )
    kde_len_test = alt.Chart(stat_text_test).transform_density('Longueur données textuelles', 
                                                as_=['Longueur données textuelles', 'densité']).mark_line(color='#29A989').encode(
                                                    x='Longueur données textuelles:Q',
                                                    y=alt.Y('densité:Q', axis=alt.Axis(titleColor='#29A989', title='Densité'))
                                                )
    mixed_char_test = alt.layer(hist_len_test, kde_len_test).resolve_scale(
        y='independent'
    ).properties(width=600, height=400, title="Distribution des longueurs de données textuelles dans X_test").configure_title(anchor='middle')
    st.altair_chart(mixed_char_test)

    st.write("L'observation de la valeur médiane de la longueur des données textuelles par code type de produits permet de constater que certaines catégories sont faiblement renseignées. Par exemple, la moitié des produits des catégories 10, 40 et 2462 contiennent moins de 50 caractères.")
    stat_text_train_per_code = stat_text_train.groupby(by='prdtypecode')['Longueur données textuelles'].median()
    stat_text_train_per_code = pd.DataFrame(stat_text_train_per_code)
    stat_text_train_per_code['prdtypecode'] = stat_text_train_per_code.index.astype(str)
    stat_text_train_per_code = stat_text_train_per_code.rename(columns={"Longueur données textuelles" : "Valeur médiane des longueurs"})
    stat_text_train_per_code = stat_text_train_per_code.sort_values(by="Valeur médiane des longueurs", ascending=False)
    stat_text_train_per_code = pd.merge(left=stat_text_train_per_code, right=df_cat, how='left', left_index=True, right_index=True)

    chart_median_len_per_code = alt.Chart(stat_text_train_per_code).mark_bar().encode(
        x=alt.X('prdtypecode:N', sort=list(stat_text_train_per_code["Valeur médiane des longueurs"].values)),
        y='Valeur médiane des longueurs:Q',
        tooltip = ['prdtypecode', 'Catégorie', 'Valeur médiane des longueurs'],
        color=alt.Color('Valeur médiane des longueurs', legend=None, scale=alt.Scale(scheme='viridis', reverse=True))  # Apply Viridis color scale
        ).properties(width=600, height=400, title="Valeur médiane de la longueur des données textuelles par catégorie du jeu X_train").configure_title(anchor='middle')

    st.altair_chart(chart_median_len_per_code)


    st.header("Présence de balises html")
    st.write("La désignation et la description des produits contiennent parfois des balises html. Le graphique suivant montre le nombre de produits dont la désignation ou la description contient des balises html.")

    train_html_tags = pd.read_csv(Path(tools.DATA_VIZ_DIR + "/features/train_html_tags.csv"), index_col=0)
    test_html_tags = pd.read_csv(Path(tools.DATA_VIZ_DIR + "/features/test_html_tags.csv"), index_col=0)
    stat_train_tags = train_html_tags.sum()
    stat_test_tags = test_html_tags.sum()
    stats_tags = pd.concat([stat_train_tags, stat_test_tags], axis=1)
    stats_tags = stats_tags.rename(columns={0: "X_train", 1: "X_test"})
    stats_tags['feature'] = stats_tags.index
    stats_tags = stats_tags.melt("feature", var_name='DataFrame', value_name="Nombre")
    stats_tags.loc[stats_tags['feature']=="designation_html_tag", 'feature'] = "designation"
    stats_tags.loc[stats_tags['feature']=="description_html_tag", 'feature'] = "description"
    stats_tags['pourcentage'] = 0.
    stats_tags.loc[stats_tags['DataFrame']=='X_train', 'pourcentage'] = stats_tags.loc[stats_tags['DataFrame']=='X_train', 'Nombre'] / x_train.shape[0] * 100.
    stats_tags.loc[stats_tags['DataFrame']=='X_test', 'pourcentage'] = stats_tags.loc[stats_tags['DataFrame']=='X_test', 'Nombre'] / x_test.shape[0] * 100.
    stats_tags['pourcentage'] = stats_tags['pourcentage'].apply(convert_to_str)   
    chart_tags = alt.Chart(stats_tags).mark_bar().encode(
        x=alt.X('feature', sort=list(stats_tags.index)),
        y="Nombre",
        color='DataFrame',
        tooltip=['DataFrame', 'feature', 'Nombre', 'pourcentage']
    ).properties(width=600, height=400, title="Nombre de produits avec des balises html").configure_title(anchor='middle').interactive()
    st.altair_chart(chart_tags)
    st.write("Presque la moitié des descriptions des produits des jeux de données X_train et X_test contiennent des balises html. Celles-ci sont nettoyées lors de la phase préprocessing via les librairies de *BeautifulSoup*.")

    st.header("Données dupliquées")
    st.write("Une analyse des données des jeux de données X_train et X_test met en évidence des duplications de certaines désignations ainsi que de certaines descriptions.")
    train = x_train
    train_designation_dup = train['designation'].duplicated(keep=False)
    train_description_dup = train['description'].dropna().duplicated(keep=False)
    train = train.fillna("")
    train["both"] = train['designation'] + train['description']
    train_both_dup = train["both"].duplicated(keep=False)
    train_dup = pd.merge(left=train_designation_dup, right = train_description_dup, how='left', left_index=True, right_index=True).fillna(False)
    train_dup = pd.merge(left=train_dup, right = train_both_dup, how='left', left_index=True, right_index=True).fillna(False)
    test = x_test
    test_designation_dup = test['designation'].duplicated(keep=False)
    test_description_dup = test['description'].dropna().duplicated(keep=False)
    test = test.fillna("")
    test["both"] = test['designation'] + test['description']
    test_both_dup = test["both"].duplicated(keep=False)
    test_dup = pd.merge(left=test_designation_dup, right = test_description_dup, how='left', left_index=True, right_index=True).fillna(False)
    test_dup = pd.merge(left=test_dup, right = test_both_dup, how='left', left_index=True, right_index=True).fillna(False)
    df_dup = pd.concat([train_dup.sum(), test_dup.sum()], axis=1)
    df_dup = df_dup.rename(columns={0: "X_train", 1: "X_test"})
    df_dup['feature'] = df_dup.index
    df_dup = df_dup.melt("feature", var_name='DataFrame', value_name="Nombre")
    df_dup['pourcentage'] = 0.
    df_dup.loc[df_dup['DataFrame']=='X_train', 'pourcentage'] = df_dup.loc[df_dup['DataFrame']=='X_train', 'Nombre'] / x_train.shape[0] * 100.
    df_dup.loc[df_dup['DataFrame']=='X_test', 'pourcentage'] = df_dup.loc[df_dup['DataFrame']=='X_test', 'Nombre'] / x_test.shape[0] * 100.
    df_dup['pourcentage'] = df_dup['pourcentage'].apply(convert_to_str)   
    chart_dup = alt.Chart(df_dup).mark_bar().encode(
        x=alt.X('feature', sort=list(df_dup.index)),
        y="Nombre",
        color='DataFrame',
        tooltip=['DataFrame', 'feature', 'Nombre', 'pourcentage']
    ).properties(width=600, height=400, title="Nombre de données dupliquées").configure_title(anchor='middle')#.interactive()
    st.altair_chart(chart_dup)

    st.write("Un croisement des données permet de constater que :")
    st.markdown("- des désignations de produits du jeu X_test sont présentes dans X_train ;")
    st.markdown("- des descriptions de produits du jeu X_test sont présentes dans X_train ;")
    st.markdown("- la désignation + la description de produits du jeu X_test sont présentes dans X_train.")

    df_test_dup_in_train = pd.read_csv(Path(tools.DATA_VIZ_DIR + "/features/x_test_data_in_train.csv"), index_col=0)
    #st.write(df_test_dup_in_train.head())
    stats_test_dup_in_train = df_test_dup_in_train.sum()
    stats_test_dup_in_train = pd.DataFrame(stats_test_dup_in_train).rename(columns={0 : "Nb duplications"})
    stats_test_dup_in_train['pourcentage'] = stats_test_dup_in_train['Nb duplications'] / x_test.shape[0] * 100.
    stats_test_dup_in_train['pourcentage'] = stats_test_dup_in_train['pourcentage'].apply(convert_to_str) 
    stats_test_dup_in_train['feature'] = stats_test_dup_in_train.index
    #st.write(stats_test_dup_in_train.head())
    chart_test_in_train = alt.Chart(stats_test_dup_in_train).mark_bar().encode(
        x=alt.X('feature', sort=list(df_dup.index)),
        y="Nb duplications",
        color=alt.Color('feature', legend=None, scale=alt.Scale(scheme='viridis', reverse=True)),#'feature',
        tooltip=['feature', 'Nb duplications', 'pourcentage']
    ).properties(width=600, height=400, title="Nombre de données de X_test présentes dans X_train").configure_title(anchor='middle')#.interactive()
    st.altair_chart(chart_test_in_train)
    st.write("La section ''Données suspectes'' étudie plus en détails les données dupliquées dans le jeu X_train et met en évidence que des produits avec la même désignation et la même description n'ont pas nécessairement le même code type attribué.")
    
    st.header("Les tokens les plus fréquents")
    st.write("En phase exploratoire du projet, une première tokenization (avec les librairies de *spacy*) a été réalisée sur la fusion de la désignation et la description du jeu X_train (après nettoyage des balises html). Une lemmatisation a été également appliquée ainsi qu'un filtrage des ''stop words''. Les caractères spéciaux ont été également filtrés et les accents convertis en caractère unicode. Le texte résultant a été converti en minuscules.")
    st.write("Cette phase de tokenisation a généré plus de 99k tokens sur le jeu X_train, dont 54% des tokens n'ont été identifiés qu'une seule fois dans les données textuelles. Cela est du à des problèmes de segmentation des mots relatifs à des oublis de caractères espace dans la saisie des désignations et descriptions.")

    with open(Path(tools.DATA_VIZ_DIR + "/features/X_train_lf0_hf27_V2dict_tokens_nb.pkl"), "rb") as f:
        dict_tokens_nb = pickle.load(f)

    df_tokens_nb = pd.DataFrame.from_dict(dict_tokens_nb, orient='index', columns=['nb_prd'])

    top_200tokens = df_tokens_nb.sort_values(by="nb_prd", ascending=False).head(200)
    top_200tokens.head()
    
    text = " ".join(top_200tokens.index)
    fig = plt.figure()
    wordc = WordCloud(background_color="rgba(255, 255, 255, 0)").generate(text)
    plt.imshow(wordc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    st.pyplot(fig)

    st.write("La langue majoritaire dans les jeux de données est le français. D'autres langues occidentales sont présentes (anglais, allemand, italien, espagnol) mais également d'autres langues orientales (japonais, chinois et même philippin). Etant donné qu'une désignation ou une description peut être composée en plusieurs langues, les statistiques obtenues sur les langues des données textuelles ne sont pas fiables.")



def explo_suspectdata(X, df_cat, df_stats, y_train):
    st.header("Données suspectes")
    st.write("L'exploration des données textuelles a mise en évidence des désignations et/ou des descriptions dupliquées dans les données textuelles. Il arrive même que plusieurs produits partagent la même désignation et la même description à la lettre près.")
    st.write("Cette partie s'intéresse à ces produits et à leurs codes types affectés.")
    st.subheader("Duplication des features (designation+description)")  
    st.write("Deux produits partageant la même désignation et la même description devraient être associés au même code type de produit.")
    st.write("Le graphique suivant montre le nombre de produits par code type partageant la même feature (désignation + description) avec au moins un autre produit qui n'a pas le même code type.")
    train_feat_dup = pd.read_csv(Path(tools.DATA_VIZ_DIR + "/features/train_feat_dup.csv"), index_col=0)
    anomalies = train_feat_dup.loc[train_feat_dup['nb_labels']>1]
    anomalies_per_code = pd.DataFrame(anomalies.groupby(by='prdtypecode')['productid'].count().sort_values(ascending=False))
    anomalies_per_code = pd.merge(left=anomalies_per_code, right=df_cat, how='left', left_index=True, right_index=True)
    anomalies_per_code['prdtypecode'] = anomalies_per_code.index
    anomalies_per_code = anomalies_per_code.rename(columns={'productid' : 'Nombre'}) 
    anomalies_per_code = pd.merge(left=anomalies_per_code, right=df_stats['count'], how='left', left_index=True, right_index=True)
    anomalies_per_code['pourcentage'] = anomalies_per_code['Nombre'] / anomalies_per_code['count'] * 100.
    anomalies_per_code['pourcentage'] = anomalies_per_code['pourcentage'].apply(convert_to_str)
    chart_ano_feat = alt.Chart(anomalies_per_code).mark_bar().encode(
        x=alt.X('prdtypecode:N', sort=list(anomalies_per_code['prdtypecode'].values)),
        y='Nombre:Q',
        tooltip = ['prdtypecode', 'Catégorie', 'Nombre', 'pourcentage'],
        color=alt.Color('Nombre', scale=alt.Scale(scheme='viridis'))  # Apply Viridis color scale
        ).properties(width=600, height=400)
    st.altair_chart(chart_ano_feat, use_container_width=True)
    list_features = list(anomalies['feature'].unique())
    anomalies['Filepath'] = anomalies.apply(lambda row : tools.get_filepath_train(row['productid'], row['imageid']), axis=1)

    selected_feature = st.selectbox('Choisir une feature pour afficher les produits associés : ', list_features)
    df_select = anomalies.loc[anomalies['feature']==selected_feature]
    list_prdtypecode = list(df_select['prdtypecode'].values)
    st.dataframe(df_cat.iloc[df_cat.index.isin(list_prdtypecode)].head(),
                column_config={
                    "prdtypecode" : st.column_config.NumberColumn("prdtypecode", format="%d")
                })

    nb_cols = 2
    nb_rows = int(np.ceil(df_select.shape[0] / nb_cols))

    fig = plt.figure()
    for i, index, prdtypecode, filepath in zip(range(df_select.shape[0]), df_select.index, df_select['prdtypecode'].values, df_select['Filepath'].values):
        ax = fig.add_subplot(nb_rows, nb_cols, i+1)
        img = plt.imread(filepath)
        ax.imshow(img)
        caption = str(prdtypecode)
        ax.set_title(f"Prd index = {index} - prdtypecode = {prdtypecode}", fontsize=6)
        ax.axis('off')
    st.pyplot(fig)

    st.write("Pour chaque produit avec une feature dupliqué, une vérification si son code type est différent de la modalité la plus commune pour les autres produits avec la même feature permet de faire une première estimation du nombre de labels suspects.")
    nb_suspect_labels = anomalies.loc[anomalies['prdtypecode']!=anomalies['Common Label']].shape[0]
    st.write("*Le nombre de codes types suspects est de ", str(nb_suspect_labels), ", soit environ %.2f%% du jeu de données y_train.*" %(nb_suspect_labels / y_train.shape[0] * 100.))
    st.write("A ce stade, le nombre de données suspectes reste négligable mais montre que les données ne sont pas complètement fiables.")

    st.subheader("Duplication des désignations")
    st.write("Une démarche similaire est réalisée sur les produits dont seulement la désignation est dupliquée.")
    st.write("Le graphique suivant montre le nombre de produits partageant la même désignation que d'autres produits mais n'ayant pas tous le même code type.")

    train_desi_dup = pd.read_csv(Path(tools.DATA_VIZ_DIR + "/features/train_desi_dup.csv"), index_col=0)
    
    anomalies_desi_per_code = pd.DataFrame(train_desi_dup.groupby(by='prdtypecode')['productid'].count().sort_values(ascending=False))
    anomalies_desi_per_code = pd.merge(left=anomalies_desi_per_code, right=df_cat, how='left', left_index=True, right_index=True)
    anomalies_desi_per_code['prdtypecode'] = anomalies_desi_per_code.index
    anomalies_desi_per_code = anomalies_desi_per_code.rename(columns={'productid' : 'Nombre'}) 
    anomalies_desi_per_code = pd.merge(left=anomalies_desi_per_code, right=df_stats['count'], how='left', left_index=True, right_index=True)
    anomalies_desi_per_code['pourcentage'] = anomalies_desi_per_code['Nombre'] / anomalies_desi_per_code['count'] * 100.
    anomalies_desi_per_code['pourcentage'] = anomalies_desi_per_code['pourcentage'].apply(convert_to_str)
#    st.write(anomalies_desi_per_code.head())
    chart_ano_desi = alt.Chart(anomalies_desi_per_code).mark_bar().encode(
        x=alt.X('prdtypecode:N', sort=list(anomalies_desi_per_code['prdtypecode'].values)),
        y='Nombre:Q',
        tooltip = ['prdtypecode', 'Catégorie', 'Nombre', 'pourcentage'],
        color=alt.Color('Nombre', scale=alt.Scale(scheme='viridis'))  # Apply Viridis color scale
        ).properties(width=600, height=400).interactive()
    st.altair_chart(chart_ano_desi, use_container_width=True)

    list_designations = list(train_desi_dup['designation'].unique())
    train_desi_dup['Filepath'] = train_desi_dup.apply(lambda row : tools.get_filepath_train(row['productid'], row['imageid']), axis=1)
    selected_designation = st.selectbox('Choisir une désignation pour afficher les produits associés : ', list_designations)    
    df_select_desi = train_desi_dup.loc[train_desi_dup['designation']==selected_designation]
    list_prdtypecode = list(df_select_desi['prdtypecode'].values)
    st.dataframe(df_cat.iloc[df_cat.index.isin(list_prdtypecode)].head(),
                column_config={
                    "prdtypecode" : st.column_config.NumberColumn("prdtypecode", format="%d")
                })
    nb_cols = 2
    nb_rows = int(np.ceil(df_select_desi.shape[0] / nb_cols))
    fig = plt.figure()
    for i, index, prdtypecode, filepath in zip(range(df_select_desi.shape[0]), df_select_desi.index, df_select_desi['prdtypecode'].values, df_select_desi['Filepath'].values):

        ax = fig.add_subplot(nb_rows, nb_cols, i+1)
        img = plt.imread(filepath)
        ax.imshow(img)
        caption = str(prdtypecode)
        ax.set_title(f"Prd index = {index} - prdtypecode = {prdtypecode}", fontsize=6)
        ax.axis('off')
    st.pyplot(fig)
    st.write("La consultation manuelle des produits partageant une même désignation permet d'identifier également des anomalies concernant des codes types affectés.")
    st.write("Pour chaque produit avec une désignation dupliqué, une comparaison entre son code type et la modalité la plus commune pour les autres produits avec la même désignation est réalisée.")
    nb_suspect_desi_labels = train_desi_dup.loc[train_desi_dup['prdtypecode']!=train_desi_dup['Common Label']].shape[0]
    st.write("*Le nombre de codes types suspects est de ", str(nb_suspect_desi_labels), ", soit environ %.2f%% du jeu de données y_train.*" %(nb_suspect_desi_labels / y_train.shape[0] * 100.))
    
    st.subheader("Duplication des descriptions")
    st.write("La même démarche est appliquée sur la description des produits après un nettoyage des balises html (fortement présentes dans les descriptions). Seuls les produits dont la longueur de la description dépasse 5 caractères sont conservés dans cette analyse.")
    st.write("Le graphique suivant montre le nombre de produits partageant la même desciption que d'autres produits mais n'ayant pas tous le même code type.")
    train_descr_dup = pd.read_csv(Path(tools.DATA_VIZ_DIR + "/features/train_descr_dup.csv"), index_col=0)
    #st.write(train_descr_dup.head())
    anomalies_descr_per_code = pd.DataFrame(train_descr_dup.groupby(by='prdtypecode')['productid'].count().sort_values(ascending=False))
    anomalies_descr_per_code = pd.merge(left=anomalies_descr_per_code, right=df_cat, how='left', left_index=True, right_index=True)
    anomalies_descr_per_code['prdtypecode'] = anomalies_descr_per_code.index
    anomalies_descr_per_code = anomalies_descr_per_code.rename(columns={'productid' : 'Nombre'}) 
    anomalies_descr_per_code = pd.merge(left=anomalies_descr_per_code, right=df_stats['count'], how='left', left_index=True, right_index=True)
    anomalies_descr_per_code['pourcentage'] = anomalies_descr_per_code['Nombre'] / anomalies_descr_per_code['count'] * 100.
    anomalies_descr_per_code['pourcentage'] = anomalies_descr_per_code['pourcentage'].apply(convert_to_str)
    #st.write(anomalies_descr_per_code.head())
    chart_ano_descr = alt.Chart(anomalies_descr_per_code).mark_bar().encode(
        x=alt.X('prdtypecode:N', sort=list(anomalies_descr_per_code['prdtypecode'].values)),
        y='Nombre:Q',
        tooltip = ['prdtypecode', 'Catégorie', 'Nombre', 'pourcentage'],
        color=alt.Color('Nombre', scale=alt.Scale(scheme='viridis'))  # Apply Viridis color scale
        ).properties(width=600, height=400).interactive()
    st.altair_chart(chart_ano_descr, use_container_width=True)    

    list_descriptions = list(train_descr_dup['description'].unique())
    train_descr_dup['Filepath'] = train_descr_dup.apply(lambda row : tools.get_filepath_train(row['productid'], row['imageid']), axis=1)
    selected_description = st.selectbox('Choisir une description pour afficher les produits associés : ', list_descriptions)    
    df_select_desc = train_descr_dup.loc[train_descr_dup['description']==selected_description]
    list_prdtypecode = list(df_select_desc['prdtypecode'].values)
    st.dataframe(df_cat.iloc[df_cat.index.isin(list_prdtypecode)].head(),
                column_config={
                    "prdtypecode" : st.column_config.NumberColumn("prdtypecode", format="%d")
                })
    nb_cols = 2
    nb_rows = int(np.ceil(df_select_desc.shape[0] / nb_cols))
    fig = plt.figure()
    for i, index, prdtypecode, filepath in zip(range(df_select_desc.shape[0]), df_select_desc.index, df_select_desc['prdtypecode'].values, df_select_desc['Filepath'].values):

        ax = fig.add_subplot(nb_rows, nb_cols, i+1)
        img = plt.imread(filepath)
        ax.imshow(img)
        caption = str(prdtypecode)
        ax.set_title(f"Prd index = {index} - prdtypecode = {prdtypecode}", fontsize=6)
        ax.axis('off')
    st.pyplot(fig)
    st.write("Une consultation manuelle des produits partageant une même description mais pas le même code type met également en évidence des incohérences sur les labels.")
    nb_suspect_desc_labels = train_descr_dup.loc[train_descr_dup['prdtypecode']!=train_descr_dup['Common Label']].shape[0]
    st.write("*Le nombre de codes types suspects est de ", str(nb_suspect_desc_labels), " sur la base de leur description, soit environ %.2f%% du jeu de données y_train.*" %(nb_suspect_desc_labels / y_train.shape[0] * 100.))
    df_union = pd.concat([train_desi_dup, train_descr_dup], axis=0)
    total_suspects = df_union.loc[df_union['prdtypecode']!=df_union['Common Label']].shape[0]
    st.write("En considérant les cas suspects détectés sur la base des désignations et des descriptions, cela représente ", str(total_suspects)," produits, soit environ %.2f%% du jeu de données y_train." % (total_suspects / y_train.shape[0] * 100.))
    st.subheader("Autres cas suspects")
    st.write("Au cours des analyses des prédictions erronées des modèles, quelques cas suspects ont également été mises en évidence. Ceux-ci ne sont pas faciles à détecter puisqu'ils ne partagent aucune donnée commune (la description, désignation et image sont différentes).")
    st.write("Ces cas sont difficilement détectables de manière automatisée puisqu'ils ne partagent aucune donnée en commun. Toutefois, voici un exemple de 2 produits qui sont très fortement similaires mais classés dans 2 catégories différentes :")
    specific_cases = X.iloc[X.index.isin([133, 32844])]
    specific_cases['Filepath'] = specific_cases.apply(lambda row : tools.get_filepath_train(row['productid'], row['imageid']), axis=1)
    specific_cases = pd.merge(left=specific_cases, right=df_cat, how='left', left_on='prdtypecode', right_index=True)
    
    cols = cycle(st.columns(2))
    for index, prdtypecode, categorie, img, designation in zip(specific_cases.index, 
                                                            specific_cases['prdtypecode'].values, 
                                                            specific_cases['Catégorie'].values, 
                                                            specific_cases['Filepath'].values, 
                                                            specific_cases['designation'].values):
        title = str("Prd Index : ") + str(index) + "  \nCode type : " + str(prdtypecode) + " \nCatégorie : " + categorie + "  \nDesignation : " + designation
        next(cols).image(img, width=350, caption=title)
    st.write("En conclusion, il est difficile d'évaluer précisément le nombre de labels suspects dans le jeu de données. Toutefois, il est acquis que certains labels ne sont pas fiables et peuvent impacter l'entraînement des modèles et la qualité des prédictions.")