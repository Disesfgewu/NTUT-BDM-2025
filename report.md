班級: 電資四 學號: 111820006 姓名: 陳羿錦

班級: 資訊四 學號: 111590009 姓名: 陳世昂


--------------------------------------------------------------------------------


Music Recommendation System: Final Project Report

1. Project Objectives and Methodology

This section outlines the project's original goals and the technical approaches selected for implementation, as defined in the initial proposal. The objective is to establish a clear baseline against which the final empirical results and conclusions can be evaluated, providing a structured, scientific assessment of the project's outcomes.

The core objective of this project was to build a music recommendation system capable of calculating song similarity and providing relevant suggestions. To achieve this, the project proposed the implementation and comparative analysis of three distinct and widely recognized recommendation algorithms.

The three primary algorithms selected for this study were:

1. Content-Based Recommendation This method operates on the principle of recommending items that are similar to those a user has liked in the past. The implementation planned to represent each song using a combination of its intrinsic audio features (e.g., tempo, loudness) and textual metadata (e.g., artist, genre). Similarity was to be calculated using metrics such as cosine similarity for audio vectors and TF-IDF or Jaccard similarity for textual attributes.
2. Collaborative Filtering (CF) This approach makes recommendations based on patterns of agreement between users. The project planned to test both user-user and item-item collaborative filtering. These techniques function by identifying users with similar listening histories or songs that are frequently listened to by the same group of users, respectively, and then recommending items based on these neighborhood patterns.
3. Latent Factor Model (Matrix Factorization) This model aims to uncover the hidden (latent) features that explain observed user-song interactions. By decomposing the user-song interaction matrix into lower-dimensional user-feature and item-feature matrices, the model learns these latent vectors. The training process was designed to use Stochastic Gradient Descent (SGD) with regularization to minimize prediction error and prevent overfitting.

The viability of these distinct algorithms, however, is entirely contingent on the underlying data. The following section, therefore, dissects the statistical properties of the Million Song Dataset to establish the specific challenges—namely sparsity and distribution skew—that each model must overcome.

2. Dataset Analysis: The Million Song Dataset

A strategic analysis of the dataset is essential for building effective models and correctly interpreting their performance. Understanding the inherent properties, distributions, and biases within the Million Song Dataset is critical, as these factors directly influence the behavior and efficacy of each recommendation algorithm.

2.1. Dataset Scale and Sparsity

The dataset's massive scale presents significant computational challenges, which are compounded by its extreme sparsity. The primary statistics are as follows:

* Total Unique Users: 1,019,310
* Total Unique Songs: 310,725
* Total Interactions (Plays): 38,603,293
* Data Sparsity: 99.9878%

This level of sparsity indicates that the vast majority of possible user-song interactions are absent from the data, making it incredibly difficult to find overlapping behavior patterns.

2.2. User and Song Distribution

The dataset exhibits highly skewed distributions in user activity, song popularity, and individual play counts, creating a classic "long-tail" problem.

* User Activity: The user base is heavily skewed towards low-activity listeners. A total of 482,832 users fall into the "Low" activity category, representing the largest single segment of the user population.
* Song Popularity: The song catalog is dominated by unpopular or "Niche" tracks. There are 117,320 songs in this category, far outnumbering the "Popular" and "Hit" songs combined.
* Play Count: The data displays an extreme long-tail phenomenon where most interactions are fleeting. Fully 59.42% of all user-song interactions in the dataset consist of only a single play.

2.3. Content Imbalance and Concentration

Further analysis reveals significant imbalances in how content is represented and consumed within the dataset.

* Genre Discrepancy: There is a notable mismatch between the number of songs in a genre and the total engagement that genre receives. For example, Rock is the most represented genre with 136,776 songs, but Jazz commands the highest total engagement with 30,242,512 plays. This divergence illustrates a critical modeling pitfall: a system naive to engagement metrics might over-represent the vast but less-played Rock catalog, while failing to capture the deep, concentrated user interest within the smaller Jazz library.
* Concentration of Popularity: A small number of artists and songs account for a disproportionate amount of listening activity. The top artist, Artist_SOBONKR1, single-handedly amassed 578,993 total plays. Furthermore, the top of the popular songs chart is dominated first by the Hip-Hop genre, followed by a large number of songs from the Blues genre.

These characteristics—extreme sparsity, severe distribution skews, and concentrated engagement—collectively define a high-difficulty benchmark. They predict a challenging environment for user-centric models and favor algorithms robust to a sparse, item-driven landscape, a hypothesis the following empirical results will test.

3. Empirical Results: Model Performance Comparison

To empirically validate the impact of the dataset's challenging characteristics, each model was subjected to a rigorous Leave-One-Out evaluation. This method involves, for each user in the test set, holding out one song from their listening history and tasking the model with recommending it back based on the remainder of their history. This evaluation was conducted on a sample population of 5,000 users. The primary metric for comparison across all models is Hit Rate@10, which measures the percentage of users for whom the held-out song was successfully recommended within their top 10 suggestions.

3.1. Overall Performance Metrics

The comprehensive results from the large-scale evaluation are synthesized in the table below. The primary metric, Hit Rate@10, is bolded for emphasis. The top-performing models for accuracy and coverage are also highlighted.

Model (Model)	Successfully Evaluated Users	Hit Rate@5	Hit Rate@10	Hit Rate@20	Coverage (Coverage)
Hybrid (混合)	5,000	0.98%	1.40%	1.96%	16.10%
Item-CF (項目協同過濾)	5,000	0.52%	1.16%	2.22%	24.27%
Content-Based (基於內容)	5,000	0.16%	0.40%	0.70%	16.23%
Matrix Factorization (MF)	4,993	0.00%	0.00%	0.00%	10.87%
User-CF (用戶協同過濾)	4,945	0.00%	0.00%	0.00%	11.99%

3.2. In-depth Model Analysis

Content-Based Recommendation

Quantitatively, the Content-Based model delivered low performance, achieving a Hit Rate@10 of only 0.40%. However, this metric belies its qualitative strength. In practical tests, the model proved highly effective at identifying and recommending songs with high feature similarity, such as consistently suggesting other Hip-Hop tracks for a given Hip-Hop song. This capability makes it valuable for addressing the "cold-start" problem, where no user interaction data is available for a new song.

Collaborative Filtering (CF)

The two CF approaches yielded starkly different results. Item-CF emerged as the best-performing single algorithm, with a Hit Rate@10 of 1.16% and the highest song coverage at 24.27%. In contrast, User-CF completely failed. With a data sparsity of 99.9878%, the probability of finding two users with a meaningful overlap in listening history is statistically negligible. This directly explains User-CF's complete failure (0.00% Hit Rate), as its foundational assumption—the existence of discoverable user 'neighbors'—is invalidated by the data's structure.

Matrix Factorization (MF)

Like User-CF, the Matrix Factorization model failed the quantitative evaluation with a Hit Rate@10 of 0.00%. This failure is attributable to the model's inability to learn meaningful user and item vectors from a user-item matrix that is 99.9878% empty. Without sufficient data points per user, the SGD training process could not converge on generalizable latent features. Despite this, its theoretical advantage was visible in qualitative tests. For a user with a preference for Classical music, the MF model recommended a diverse set of genres including Jazz, Folk, and Rock, demonstrating a potential for personalized discovery that other models lacked, even if it failed the strict accuracy test.

Hybrid Recommender

The Hybrid model was the overall top performer, achieving the highest Hit Rate@10 of 1.40%. Its success is attributed to its ability to combine the outputs of the individual models using a weighted strategy (e.g., assigning a 30% weight to the Content-Based model). This approach effectively balances the strengths and weaknesses of each component, leveraging the feature-matching of Content-Based filtering and the collaborative signal of Item-CF to produce a more robust and accurate final recommendation list.

These empirical results consistently underscore the decisive role that data sparsity plays in shaping model performance and determining the viability of different algorithmic approaches.

4. Final Conclusion

This project began with the clear objective of implementing and comparing three foundational recommendation methods—Content-Based, Collaborative Filtering, and Matrix Factorization—on the Million Song Dataset. It culminated in a large-scale empirical study that not only measured the quantitative performance of these algorithms but also revealed their practical behaviors and limitations when faced with real-world data challenges.

The central and unequivocal finding of this project is that the extreme data sparsity (99.9878%) of the Million Song Dataset was the single most dominant factor influencing all outcomes. This characteristic profoundly handicapped algorithms that depend on rich, overlapping user data and dictated the relative success of the different approaches.

The key takeaways from the model comparison can be summarized as follows:

* Overall Low Accuracy: The inherent difficulty of the task is reflected in the low absolute performance metrics. Even the best-performing hybrid model achieved a Hit Rate@10 of only 1.40%, highlighting the challenge of making accurate predictions in such a sparse environment.
* Invalidation of User-Centric Models: Algorithms predicated on rich user-behavior patterns, namely User-CF and Matrix Factorization, were rendered ineffective by the dataset's sparsity. The statistical near-impossibility of finding users with meaningful listening overlap or learning generalizable latent features from an almost-empty matrix led to their complete failure in the evaluation.
* Relative Robustness of Item-Centric Methods: In contrast, models that focus on item-item relationships (Item-CF) or intrinsic item features (Content-Based) proved more resilient to data sparsity. Item-CF was the most effective standalone model.
* Superiority of Hybrid Approach: The hybrid model achieved the highest accuracy by strategically combining the outputs of its constituent models. This approach successfully mitigated the individual weaknesses of each algorithm, leading to a more balanced and effective recommendation engine.

Ultimately, the project was successful in meeting its original objectives. It systematically implemented and rigorously compared the proposed algorithms, providing a clear demonstration of their practical limitations and behaviors when confronted with the formidable challenges posed by a sparse, massive, and long-tailed real-world dataset.
