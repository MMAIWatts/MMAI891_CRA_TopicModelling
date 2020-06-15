####################
###Import packages##
####################

import matplotlib.pyplot as plt
import gensim
import numpy as np
import pandas as pd
from gensim.models import CoherenceModel
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
import os
from collections import Counter
import seaborn as sns
sns.set()
# t-SNE Clustering Chart
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook
# from bokeh.io import export_png
from tqdm import tqdm

# pyLDAVis
import pyLDAvis.gensim

# WordCloud
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

from parameters import *
from NLP_preprocessing import *



## LDA function ###
def lda(num_topics, corpus, dictionary):
    # Build LDA model
    model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=dictionary,
                                            num_topics=num_topics,
                                            random_state=42,
                                            update_every=1,
                                            chunksize=10,
                                            passes=10,
                                            alpha='symmetric',
                                            iterations=100,
                                            per_word_topics=True)

    return model


# coherence vs number of topics plot -- run it once after preprocessing to choose the proper number of topics
def coherence_vs_topics(texts, dictionary, corpus, min_number_topics=5, max_number_topics=15):
    scores = []
    topics_range = tqdm(range(min_number_topics, max_number_topics + 1))
    for n_topic in topics_range:
        ldamodel = lda(num_topics=n_topic, corpus=corpus, dictionary=dictionary)
        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=ldamodel, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        scores.append(coherence_lda)
    
    trange = range(min_number_topics, max_number_topics + 1)
    pd.DataFrame({'Number of Topics':trange, 'Coherence Score': scores}).to_csv(os.path.join(OUT_DIR, 'coherence_Topics_score.csv'))
    plt.figure(figsize = (6,4))
    plt.plot(trange, scores)
    plt.xticks(trange)
    plt.xlabel('Number of Topics')
    plt.ylabel('coherence score')
    plt.savefig(os.path.join(OUT_DIR, 'coherence_Topics_plot.png'))
    plt.show()

    

##Some functions for Topic analysis##
def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(
                    pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
            else:
                break

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    sent_topics_df.columns = ['Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    return (sent_topics_df)



# Sentence Coloring of N Sentences
def sentences_chart(ldamodel, corpus, start=0, end=12):
    corp = corpus[start:end]
    mycolors = [color for name, color in mcolors.XKCD_COLORS.items()]

    fig, axes = plt.subplots(end - start, 1, figsize=(20, (end - start) * 0.95), dpi=160)
    axes[0].axis('off')
    for i, ax in enumerate(axes):
        if i > 0:
            corp_cur = corp[i - 1]
            topic_percs, wordid_topics, wordid_phivalues = ldamodel[corp_cur]
            word_dominanttopic = [(ldamodel.id2word[wd], topic[0]) for wd, topic in wordid_topics]
            ax.text(0.01, 0.5, "Doc " + str(i - 1) + ": ", verticalalignment='center',
                    fontsize=16, color='black', transform=ax.transAxes, fontweight=700)

            # Draw Rectange
            topic_percs_sorted = sorted(topic_percs, key=lambda x: (x[1]), reverse=True)
            ax.add_patch(Rectangle((0.0, 0.05), 0.99, 0.90, fill=None, alpha=1,
                                   color=mycolors[topic_percs_sorted[0][0]], linewidth=2))

            word_pos = 0.06
            for j, (word, topics) in enumerate(word_dominanttopic):
                if j < 14:
                    ax.text(word_pos, 0.5, word,
                            horizontalalignment='left',
                            verticalalignment='center',
                            fontsize=16, color=mycolors[topics],
                            transform=ax.transAxes, fontweight=700)
                    word_pos += .009 * len(word)  # to move the word for the next iter
                    ax.axis('off')
            ax.text(word_pos, 0.5, '. . .',
                    horizontalalignment='left',
                    verticalalignment='center',
                    fontsize=16, color='black',
                    transform=ax.transAxes)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle('Sentence Topic Coloring for Documents: ' + str(start) + ' to ' + str(end - 2), fontsize=22, y=0.95,
                 fontweight=700)
    plt.tight_layout()
    plt.show()


    
def topics_per_document(ldamodel, corpus, start=0, end=-1):
    corpus_sel = corpus[start:end]
    dominant_topics = []
    topic_percentages = []
    for i, corp in enumerate(corpus_sel):
        topic_percs, wordid_topics, wordid_phivalues = ldamodel[corp]
        dominant_topic = sorted(topic_percs, key=lambda x: x[1], reverse=True)[0][0]
        dominant_topics.append((i, dominant_topic))
        topic_percentages.append(topic_percs)

    # Distribution of Dominant Topics in Each Document
    df = pd.DataFrame(dominant_topics, columns=['Document_Id', 'Dominant_Topic'])
    dominant_topic_in_each_doc = df.groupby('Dominant_Topic').size()
    df_dominant_topic_in_each_doc = dominant_topic_in_each_doc.to_frame(name='count').reset_index()

    # Total Topic Distribution by actual weight
    topic_weightage_by_doc = pd.DataFrame([dict(t) for t in topic_percentages])
    df_topic_weightage_by_doc = topic_weightage_by_doc.sum().to_frame(name='count').reset_index()

    return df_dominant_topic_in_each_doc, df_topic_weightage_by_doc



# Plot Topic Distribution by Dominant Topics
def Plot_topic_dist(ldamodel, num_topics, corpus):
    topic_top3words = [(i, topic) for i, topics in ldamodel.show_topics(num_topics, num_words=20, formatted=False)
                       for j, (topic, wt) in enumerate(topics) if j < 3]

    df_top3words_stacked = pd.DataFrame(topic_top3words, columns=['topic_id', 'words'])
    df_top3words = df_top3words_stacked.groupby('topic_id').agg(', \n'.join)
    df_top3words.reset_index(level=0, inplace=True)

    df_dominant_topic_in_each_doc, df_topic_weightage_by_doc = topics_per_document(ldamodel, corpus, start=0, end=-1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), dpi=100, sharey=True)
    plt.subplots_adjust(hspace=0.7)
    # Topic Distribution by Dominant Topics
    ax1.bar(x='Dominant_Topic', height='count', data=df_dominant_topic_in_each_doc, width=.5, color='firebrick')
    ax1.set_xticks(range(df_dominant_topic_in_each_doc.Dominant_Topic.unique().__len__()))
    tick_formatter = FuncFormatter(
        lambda x, pos: 'Topic ' + str(x) + '\n' + df_top3words.loc[df_top3words.topic_id == x, 'words'].values[0])
    ax1.xaxis.set_major_formatter(tick_formatter)
    ax1.set_title('Number of Documents by Dominant Topic', fontdict=dict(size=10))
    ax1.set_ylabel('Number of Documents')
    # ax1.set_ylim(0, 1500)

    # Topic Distribution by Topic Weights
    ax2.bar(x='index', height='count', data=df_topic_weightage_by_doc, width=.5, color='steelblue')
    ax2.set_xticks(range(df_topic_weightage_by_doc.index.unique().__len__()))
    ax2.xaxis.set_major_formatter(tick_formatter)
    ax2.set_title('Number of Documents by Topic Weightage', fontdict=dict(size=10))
    ax2.set_ylabel('Number of Documents')

    plt.show()

    

# Word Counts of Topic Keywords
#TODO change this function to be general - now works for 9 topics and y_lim is proper of one question
def word_count_topic(ldamodel, num_topics, texts):
    topics = ldamodel.show_topics(num_topics=num_topics, formatted=False)
    data_flat = [w for w_list in texts for w in w_list]
    counter = Counter(data_flat)

    out = []
    for i, topic in topics:
        for word, weight in topic:
            out.append([word, i, weight, counter[word]])

    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])

    # Plot Word Count and Weights of Topic Keywords
    fig, axes = plt.subplots(3, 5, figsize=(16, 10), sharey=False, dpi=160)
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    for i, ax in enumerate(axes.flatten()):
        ax.bar(x='word', height="word_count", data=df.loc[df.topic_id == i, :], color=cols[i], width=0.5, alpha=0.3,
               label='Word Count')
        ax_twin = ax.twinx()
        ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id == i, :], color=cols[i], width=0.2,
                    label='Weights')
        ax.set_ylabel('Word Count', color=cols[i])
        ax_twin.set_ylim(0, 0.230)
        ax.set_ylim(0, 1500)
        ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
        ax.tick_params(axis='y', left=False)
        ax.set_xticklabels(df.loc[df.topic_id == i, 'word'], rotation=30, horizontalalignment='right')
        ax.legend(loc='upper left')
        ax_twin.legend(loc='upper right')

    fig.tight_layout(w_pad=2)
    fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)
    plt.show()


    
# Wordcloud
def get_wordcloud_LDA(ldamodel, num_topics):
#     cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

    cloud = WordCloud(width=800, height=560,
                      background_color='white', collocations=False,
                      min_font_size=10)

    topics = ldamodel.show_topics(num_topics=num_topics, num_words=50, formatted=False)

    for i in range(num_topics):
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)

        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=10))
        plt.gca().axis('off')

        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout(pad=0)
        dr = '{}/wordCloud/LDA'.format(OUT_DIR)
        if not os.path.exists(dr):
            os.makedirs(dr)
        plt.savefig(os.path.join(dr, f'Topic{i}__wordcloud.png'))
        plt.show()


#########################
# - T-SNE visualization -#
#########################

# t-SNE Clustering Chart
def tsne_plot(ldamodel, corpus, num_topics,
              Keep_well_separated_pcnt=0.2):  # Keep_well_separated_pcnt = 0: keep all points
    # Get topic weights
    topic_weights = []
    for i, row_list in enumerate(ldamodel[corpus]):
        topic_weights.append([w for i, w in row_list[0]])

    # Array of topic weights    
    arr = pd.DataFrame(topic_weights).fillna(0).values
    # Keep the well separated points
    arr = arr[np.amax(arr, axis=1) > Keep_well_separated_pcnt]

    # Dominant topic number in each doc
    topic_num = np.argmax(arr, axis=1)
    
    # tSNE Dimension Reduction
    # TODO tune hyperparameters: like perplexity --can add to function arguments
    tsne_model = TSNE(n_components=2, verbose=1, random_state=42, angle=.3, init='pca')

    tsne_lda = tsne_model.fit_transform(arr)

    # Plot the Topic Clusters using Bokeh
    output_notebook()

    mycolors = np.array([color for name, color in
                         mcolors.XKCD_COLORS.items()])  # TABLEAU_COLORS (max 10 topics), XKCD_COLORS -> more than 10
    plot = figure(title="t-SNE Clustering of {} LDA Topics".format(num_topics),
                  plot_width=900, plot_height=700)
    plot.scatter(x=tsne_lda[:, 0], y=tsne_lda[:, 1], color=mycolors[topic_num])
    #     export_png(plot, filename=os.path.join(OUT_DIR , "tsne.png"))
    show(plot)


# pyLDAVis
def pyldavis_plot(ldamodel, corpus, mds="tsne"):
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary=ldamodel.id2word, mds=mds)
    return vis


# Top 3 topics per document
def top3_topics_per_document(ldamodel, corpus):
    topics_top3 = []

    for i, corp in enumerate(corpus):
        topic_percs, wordid_topics, wordid_phivalues = ldamodel[corp]
        dominant_topic = sorted(topic_percs, key=lambda x: x[1], reverse=True)[0][0]
        try:
            topic2 = sorted(topic_percs, key=lambda x: x[1], reverse=True)[1][0]
        except:
            topic2 = None        
        try:
            topic3 = sorted(topic_percs, key=lambda x: x[1], reverse=True)[2][0]
        except:
            topic3 = None
        topics_top3.append((i, dominant_topic , topic2, topic3 ))       


    df = pd.DataFrame(topics_top3, columns = ['Document_Id', 'Dominant_Topic', 'Second_Topic', 'Third_Topic'])
    
    return df

# LDA vector
def save_vec_lda(model, corpus, k):
    """
    Get the LDA vector representation (probabilistic topic assignments for all documents)
    :return: vec_lda with dimension: (n_doc * n_topic)
    """
    n_doc = len(corpus)
    vec_lda = np.zeros((n_doc, k))
    for i in range(n_doc):
        # get the distribution for the i-th document in corpus
        for topic, prob in model.get_document_topics(corpus[i]):
            vec_lda[i, topic] = prob
    
    np.savetxt(os.path.join(out_dir, '/lda_vec.csv'), vec_lda, delimiter=",")

    return



if __name__ == '__main__':
    pass
