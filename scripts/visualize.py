from google.protobuf.symbol_database import Default
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


def catplot(df,x,y,kind,title,hue=None,palette='Set3',xlabel=None,ylabel=None,size=Default):
    plt.figure(figsize=(12, 7))
    ax = sn.catplot(data=df,x=x,y=y,hue=hue,kind=kind, palette=palette,size=size)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    plt.title(title)
    plt.show()

def scatterplot(df,x,y,title,hue=None,style=None):
    plt.figure(figsize=(12, 7))
    sn.scatterplot(data=df, x=x, y=y, hue=hue, style=style)
    plt.title(title,fontsize=23)
    plt.show()

def histplot(df,x,y,title,hue=None,palette='Set2'):
    plt.figure(figsize=(12, 7))
    sn.distplot(data=df,x=x,y=y,hue=hue,palette=palette)
    plt.title(title)
    plt.show()

def correlation_heatmap(corr):
    ax = sn.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sn.diverging_palette(20, 220, n=200),
    square=True
)
    ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)

def lineplot(df,x=None,y=None,title=None,size=None,w_size=10,h_size=7):
    plt.figure(figsize=(w_size, h_size))
    sn.lineplot(data=df, x=x, y=y)
    plt.title(title)
    plt.show()

def displot(df,x=None,title=None):
    plt.figure(figsize=(12, 7))
    sn.displot(data=df, x=x)
    plt.title(title)
    plt.show()