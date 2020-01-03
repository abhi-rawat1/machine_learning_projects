# Recommendation System

Note: Below are my notes around "Recommendation System" - no need to be completely accurate.


A. Content based filtering: Based on past user transaction, show relevant items as a recommendation
	Cosine similarity - for calculating content based filtering, cosine similarity formula is used. Range is from -1 to 1

	A major drawback of this algorithm is that it is limited to recommending items that are of the same type. It will never recommend products which the user 
	has not bought or liked in the past. So if a user has watched or liked only action movies in the past, the system will recommend only action movies. 
	It’s a very narrow way of building an engine.To improve on this type of system, we need an algorithm that can recommend items not just based on the content, 
	but the behavior of users as well

B. Collaborative  filtering: To start with it, need to get an answer of below questions:
	a. How do you determine which users or items are similar to one another?
	b. Given that you know which users are similar, how do you determine the rating that a user would give to an item based on the ratings of similar users?
	c. How do you measure the accuracy of the ratings you calculate?
	
	Following are the type of collaborative filtering: 
	a. user-user filtering (when number of user < numner of item)
		For a user U, with a set of similar users determined based on rating vectors consisting of given item ratings, the rating for an item I, which hasn’t been 
		rated,is found by picking out N users from the similarity list who have rated the item I and calculating the rating based on these N ratings.
	
		Identify - 
			a. Similarity between users (using pearson algorithm)
			b. Rating of movies by users
		and then calculate prediciton of movies (i) for user (u) -> or recommendation of movies for a user 
	
	b. item-item filtering (when number of user > numner of item)
		For an item I, with a set of similar items determined based on rating vectors consisting of received user ratings, the rating by a user U, who hasn’t 
		rated it, is found by picking out N items from the similarity list that have been rated by U and calculating the rating based on these N ratings.

		Item-based collaborative filtering was developed by Amazon. In a system where there are more users than items, item-based filtering is faster and more stable 
		than user-based. It is effective because usually, the average rating received by an item doesn’t change as quickly as the average rating given by a user 
		to different items. It’s also known to perform better than the user-based approach when the ratings matrix is sparse.

		Although, the item-based approach performs poorly for datasets with browsing or entertainment related items such as MovieLens, where the 
		recommendations it gives out seem very obvious to the target users. Such datasets see better results with matrix factorization techniques, which 
		you’ll see in the next section, or with hybrid recommenders that also take into account the content of the data like the genre by using content-based 
		filtering.
		
	Sparse Matrix: when there are lots of empty valures present in matrix. In case of movie dataset, if there are lots of movies which are not rated by users 
	then Rating matrix will contain lots of empty rating cells. 

	Dense matrix: when there is very less number of empty cells

	To calculate similairty between items: 
	a. scipy.spatial.distance.euclidean  -> This algorithm calculates the distance between two points. Small distance -> Higher similairty
	b. cosine similairty

	small distance -> lower angle ->  Higher similairty
	To calculate similarity using angle, you need a function that returns a higher similarity or smaller distance for a lower angle and a lower similarity or larger distance for a higher angle. 
	The cosine of an angle is a function that decreases from 1 to -1 as the angle increases from 0 to 180.

C. Hybrid Recommender :  Its a mix of collaborative filtering and content based filtering .. 


D. Dimensionality Reduction
	In the user-item matrix, there are two dimensions:
	* The number of users
	* The number of items

	If the matrix is mostly empty, reducing dimensions can improve the performance of the algorithm in terms of both space and time. You can use various 
	methods like matrix factorization or autoencoders to do this.

E. Matrix Factorization:
	It can be seen as breaking down a large matrix into a product of smaller ones. This is similar to the factorization of integers, where 12 can be 
	written as 6 x 2 or 4 x 3. In the case of matrices, a matrix A with dimensions m x n can be reduced to a product of two matrices X and Y with dimensions 
	m x p and p x n respectively.
	
F.  Singular value decomposition (SVD) algorithm: This is one of the algorithm to compute Matrix Factorization. Other algorithm include PCA, NMF, Autoencoder etc.


Following are coupld of links -
A. https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-recommendation-engine-python/
	This has different implementations of code for Movie recommendations. It does 2 versions of collaborative filtering and ones of matrix factorization. 
	
B. https://www.analyticsvidhya.com/blog/2016/06/quick-guide-build-recommendation-engine-python/
	It is similar to the first as it has collaborative filtering but it implements a popularity algorithm.
	
C. https://beckernick.github.io/matrix-factorization-recommender/

Links:
A.  For offlien and online measurement:  https://medium.com/recombee-blog/evaluating-recommender-systems-choosing-the-best-one-for-your-business-c688ab781a35
B.  https://medium.com/recombee-blog/machine-learning-for-recommender-systems-part-1-algorithms-evaluation-and-cold-start-6f696683d0ed
C.  Different type of RS (content based, Matrix factorization, colloborative filtering etc) - https://medium.com/@madasamy/introduction-to-recommendation-systems-and-how-to-design-recommendation-system-that-resembling-the-9ac167e30e95
D.  Evaluating rs : https://towardsdatascience.com/evaluation-metrics-for-recommender-systems-df56c6611093
E.  RS blog- https://towardsdatascience.com/building-and-testing-recommender-systems-with-surprise-step-by-step-d4ba702ef80b
F.  https://beckernick.github.io/matrix-factorization-recommender/
G.  RS Evaluation :  https://towardsdatascience.com/evaluating-a-real-life-recommender-system-error-based-and-ranking-based-84708e3285b
H.  https://realpython.com/build-recommendation-engine-collaborative-filtering/#memory-based
I.  https://www.analyticsvidhya.com/blog/2016/06/quick-guide-build-recommendation-engine-python/

Example or source code:
1. https://www.kaggle.com/rounakbanik/movie-recommender-systems   - how to use hidden features in model building 
2. https://github.com/statisticianinstilettos/recmetrics/blob/master/example.ipynb   - different ways to evaluate rs  [its link with D point]
3. https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/Building%20Recommender%20System%20with%20Surprise.ipynb   [E point is linked with it]
4. https://kerpanic.wordpress.com/2018/03/26/a-gentle-guide-to-recommender-systems-with-surprise/
5. https://medium.com/@wwwbbb8510/python-implementation-of-baseline-item-based-collaborative-filtering-2ba7c8960590
6. https://github.com/wwwbbb8510/baseline-rs/blob/master/item-based-collaborative-filtering.ipynb
7. https://github.com/KevinLiao159/MyDataSciencePortfolio/tree/master/movie_recommender
	