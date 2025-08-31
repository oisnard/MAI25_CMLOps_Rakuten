import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import src.tools.tools as tools
from pathlib import Path
import altair as alt

def exploration_image(Y_train):
   st.header("Exploration des données - Images")
   st.subheader("Aire normalisée des images du jeu d'entraînement")

   X_train_area_normalized = pd.read_csv(Path(tools.DATA_VIZ_DIR+'/features/X_train_area_normalized_v2.csv'), index_col = 'Unnamed: 0')


   df = X_train_area_normalized.merge(Y_train, left_index = True, right_index = True)

   surface_mediane = df[['area_normalized', 'prdtypecode']].groupby('prdtypecode').agg({ 'area_normalized' : 'median'}).reset_index()
   surface_mediane = surface_mediane.rename(columns = {'area_normalized' : 'area_normalized_mediane'})

   surface_mediane['prdtypecode'] = surface_mediane['prdtypecode'].astype(str)

   surface_mediane = surface_mediane.sort_values(by='area_normalized_mediane', ascending=False)

   df_low_contrast = pd.read_csv(Path(tools.DATA_VIZ_DIR+'/features/IM-01-X_train_is_low_contrast.csv'), index_col = 0)
   df_low_contrast = df_low_contrast.merge(Y_train, left_index = True, right_index = True)

   df_low_contrast_mean = df_low_contrast.groupby('prdtypecode')[['prdtypecode', 'is_low_contrast']].agg({'is_low_contrast' : 'mean'}).reset_index()
   df_low_contrast_mean['prdtypecode'] = df_low_contrast_mean['prdtypecode'].astype(str)
   df_low_contrast_mean = df_low_contrast_mean.sort_values(by='is_low_contrast', ascending = False)

   df_blur = pd.read_csv(Path(tools.DATA_VIZ_DIR+'/features/X_train_blur.csv'), index_col = 0)
   df_blur = df_blur.merge(Y_train, left_index = True, right_index = True)

   df_blur_mean = df_blur.groupby('prdtypecode').agg({'blur': 'mean'}).reset_index()
   df_blur_mean['prdtypecode'] = df_blur_mean['prdtypecode'].astype(str)
   df_blur_mean = df_blur_mean.sort_values(by='blur', ascending = False)

   st.write("La distribution de l'aire normalisée sur tout le train est donnée par la boîte à moustaches suivante : ")

   chart = alt.Chart(df).mark_boxplot(size=100).encode(
      y=alt.Y("area_normalized:Q", title="Aire normalisée")
   ).properties(
      title="Distribution de l'aire normalisée",
      width=200,
      height=300
   ).configure_title(anchor="middle")
   st.altair_chart(chart, use_container_width=True)

   st.write("En regardant par catégorie de produits : ")
   chart = (
      alt.Chart(surface_mediane)
      .mark_bar()
      .encode(
         x=alt.X(
               "prdtypecode:N",
               title="Code produit",
               sort='-y',  # tri décroissant selon y
               axis=alt.Axis(labelAngle=90)
         ),
         y=alt.Y("area_normalized_mediane:Q", title="Aire normalisée médiane"),
         color=alt.Color(
               "prdtypecode:N",
               legend=alt.Legend(title="Code produit"),
               scale=alt.Scale(scheme="viridis")  # palette viridis
         ),
         tooltip=["prdtypecode", "area_normalized_mediane"]
      )
      .properties(
         title="Distribution de l'aire normalisée médiane par prdtypecode",
         width=700,
         height=450
      ).configure_title(anchor="middle")
   )

   st.altair_chart(chart, use_container_width=True)

   st.subheader("Analyse du contraste des images")
   st.write("Le nombre d'images trop faiblement contrastés est faible par rapport à la taille totale du jeu de données : ")

   chart = alt.Chart(df_low_contrast_mean).mark_bar().encode(
      x=alt.X("prdtypecode:N", title="Code produit", sort='-y', axis=alt.Axis(labelAngle=90)),
      y=alt.Y("is_low_contrast:Q", title="Proportion d'images à faible contraste"),
      color=alt.Color("prdtypecode:N", legend=alt.Legend(title="Code produit"), scale=alt.Scale(scheme="viridis")),
      tooltip=["prdtypecode", "is_low_contrast"]
   ).properties(
      title="Proportion d'images à faible contraste par code produit",
      width=700,
      height=450
   ).configure_title(anchor="middle")

   st.altair_chart(chart, use_container_width=True)   


   st.subheader("Analyse du flou des images")

   st.write("Le nombre d'images floues est donné par le graphique suivant : ")

   chart = alt.Chart(df_blur_mean).mark_bar().encode(
      x=alt.X("prdtypecode:N", title="Code produit", sort='-y', axis=alt.Axis(labelAngle=90)),
      y=alt.Y("blur:Q", title="Proportion d'images floues"),
      color=alt.Color("prdtypecode:N", legend=alt.Legend(title="Code produit"), scale=alt.Scale(scheme="viridis")),
      tooltip=["prdtypecode", "blur"]
   ).properties(
      title="Proportion d'images floues par code produit",
      width=700,
      height=450
   ).configure_title(anchor="middle")

   st.altair_chart(chart, use_container_width=True)
