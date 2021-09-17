import pandas as pd
import numpy as np
import os
pwd = os.path.abspath('.') + '\\'

class Collaborative_Filtering:
    def __init__(self, data):
        self.arr = pd.Series({user_id: pd.Series(grp['stars'].values, index=grp['business_id']).to_dict() 
                                 for user_id, grp in data.groupby(['user_id'])})
        self.length = len(data)
        #self.avuserrat = data.groupby(['user_id']).mean()['stars']
        #self.avbusrat = data.groupby(['business_id']).mean()['stars']
        
        if len(data.columns) == 4:
            self.avuserrat = pd.Series({user_id: grp['user_ave_stars'].values[0] for user_id, grp in data.groupby(['user_id'])})    
        if len(data.columns) == 5:
            self.avuserrat = pd.Series({user_id: grp['user_ave_stars'].values[0] for user_id, grp in data.groupby(['user_id'])}) 
            self.avbusrat = pd.Series({business_id: grp['bus_ave_stars'].values[0] for business_id, grp in data.groupby(['business_id'])})
    
    
    def has_shared(self, x1, x2):
        both = np.array([])
        for item in self.arr[x1]:
            if item in self.arr[x2]:
                both = np.append(both, item)
        if len(both) == 0:
            return False, [0]
        else:
            return True, both
        
    def pearson_wrong(self, x1, x2):
        shared, both = self.has_shared(x1, x2)
        n = len(both)
        sum1,sum2,prod, sqrd1,sqrd2 = 0, 0, 0, 0, 0
        if shared:
            for i in both:
                sum1 += self.arr[x1][i]
                sum2 += self.arr[x2][i]
                sqrd1 += np.power(self.arr[x1][i],2)
                sqrd2 += np.power(self.arr[x2][i],2)
                prod += self.arr[x1][i]*self.arr[x2][i]
            sum1sqrd = np.power(sum1, 2)
            sum2sqrd = np.power(sum2, 2)
            numer = (n * prod) - (sum1 * sum2)
            denom = np.sqrt(((n*sqrd1)-sum1sqrd)*((n*sqrd2)-sum2sqrd))
            if denom == 0:
                denom += .00000001
            pearson_correlation = numer/denom
            return pearson_correlation
        else:
            return 0


    def euclidean(self, x1, x2):
        distances = []
        if self.has_shared(x1, x2):
            for item in self.arr[x1]:
                if item in self.arr[x2]:
                    distances.append(pow(self.arr[x1][item] - self.arr[x2][item],2))
            distances_sum = sum(distances)
            euclidean_distance = 1/(1+np.sqrt(distances_sum))
            euclidean_distance = euclidean_distance - 1.01
            euclidean_distance = euclidean_distance * -1
            return euclidean_distance
        return 0


    def jaccard(self, x1, x2):
        shared, both = self.has_shared(x1, x2)
        left = self.arr[x1]
        right = self.arr[x2]
        union = dict(list(left.items()) + list(right.items()))
        intersect = {key: self.arr[x1][key] for key in both}
        jaccard = float(len(intersect))/len(union)
        return jaccard


    def cosine(self, x1,x2):
        shared, both = self.has_shared(x1, x2)
        numer, sqrd1, sqrd2 = 0, 0, 0
        if shared:
            for i in both:
                numer += self.arr[x1][i] * self.arr[x2][i]
                sqrd1 += self.arr[x1][i] * self.arr[x1][i]
                sqrd2 += self.arr[x2][i] * self.arr[x2][i]
            denom = np.sqrt(sqrd1) * np.sqrt(sqrd2)
            if denom == 0:
                denom += .00000001
            cosine = numer/denom
            return cosine
        else:
            return 0


    def pearsons(self, user1, user2):
        if user1 == user2:  # same
            return 1
        
        # compare user1 with user2
        rating1 = np.array([])  # rating of user1 in both
        rating2 = np.array([])  # rating of user2 in both
        for item in self.arr[user1]:
            if item in self.arr[user2]:
                rating1 = np.append(rating1, self.arr[user1][item])
                rating2 = np.append(rating2, self.arr[user2][item])
        
        length1 = len(rating1)
        length2 = len(rating2)
        if length1 == 0 or length2 == 0 or length1 != length2:  # no item in both
            return 0
        
        difference1 = np.subtract(rating1, np.mean(rating1))
        difference2 = np.subtract(rating2, np.mean(rating2))
        
        numerator = np.sum(np.multiply(difference1, difference2))
        denominator = np.sqrt(np.multiply(np.sum(np.power(difference1, 2)), np.sum(np.power(difference2, 2))))
        
        if numerator == 0 or denominator == 0:
            similarity = np.divide(np.mean(rating1), np.mean(rating2))
            return similarity if similarity <= 1 else np.divide(1, similarity)
        else:
            similarity = np.divide(numerator, denominator)
            return similarity
        
        
    def find_neighbors_of(self, person, algo = 'Pearsons', k=None): 
        scores = np.array([])
        names = np.array([])
        count = 0
        if algo == 'Pearsons':
            for user in self.arr.iteritems():
                count+=1
                if user != person:
                    comparison = self.pearsons(person, user[0])
                    scores = np.append(scores, comparison)
                    names = np.append(names, user[0])
        elif algo == 'Pearson_Wrong':
            for user in self.arr.iteritems():
                if user != person:
                    comparison = self.pearson_will(person, user[0])
                    scores = np.append(scores, comparison)
                    names = np.append(names, user[0])
        elif algo == 'Euclidean':
            for user in self.arr.iteritems():
                if user != person:
                    comparison = self.euclidean(person, user[0])
                    scores = np.append(scores, comparison)
                    names = np.append(names, user[0])
        elif algo == 'Jaccard':
            for user in self.arr.iteritems():
                if user != person:
                    comparison = self.jaccard(person, user[0])
                    scores = np.append(scores, comparison)
                    names = np.append(names, user[0])
        elif algo == 'Cosine':
            for user in self.arr.iteritems():
                if user != person:
                    comparison = self.cosine(person, user[0])
                    scores = np.append(scores, comparison)
                    names = np.append(names, user[0])
        
        similar = np.concatenate([[names, scores]]).T
        similar = sorted(similar, key=lambda row: row[1], reverse=True)
        if k is not None:
            return similar[1:k]    
        else:
            return similar[1:]
        
        
    def rating_simple(self, user, item, algo, k=None):
        neighbors = self.find_neighbors_of(user, algo, k)
        if len(neighbors) == 0:
            return self.avuserrat[user]
        
        rating = np.array([])
        for neighbor in neighbors:
            name = neighbor[0]
            if item in self.arr[name]:
                rating = np.append(rating, self.arr[name][item])
        
        if len(rating) == 0:
            return self.avuserrat[user]
        else:
            return np.mean(rating)
    
    
    def rating_weighted(self, user, item, algo, k=None):
        neighbors = self.find_neighbors_of(user, algo, k)
        if len(neighbors) == 0:
            return self.avuserrat[user]
        
        rating = np.array([])
        similarity = np.array([])
        
        for neighbor in neighbors:
            name = neighbor[0]
            score = neighbor[1]
            if item in self.arr[name]:
                rating = np.append(rating, self.arr[name][item])
                similarity = np.append(similarity, float(score))
                
        if len(rating) == 0:
            return self.avuserrat[user]
        else:
            numerator = np.sum(np.multiply(rating, similarity))
            denominator = np.sum(similarity)
            if denominator == 0:
                return self.avuserrat[user]
            prediction = np.divide(numerator, denominator)
            return prediction
        
        
    def rating_weighted_intermidiate(self, user, item, algo, k=None):
        neighbors = self.find_neighbors_of(user, algo, k)
        if len(neighbors) == 0:
            return self.avuserrat[user]
        
        difference = np.array([])
        similarity = np.array([])
        
        for neighbor in neighbors:
            name = neighbor[0]
            score = neighbor[1]
            if item in self.arr[name]:
                difference = np.append(difference, np.subtract(self.arr[name][item], self.avuserrat[name]))
                similarity = np.append(similarity, float(score))
        
        if len(difference) == 0:
            return self.avuserrat[user]
        else:
            numerator = np.sum(np.multiply(difference, similarity))
            denominator = np.sum(similarity)
            baseline = self.avuserrat[user]
            if denominator == 0:
                return baseline
            deviation = np.divide(numerator, denominator)
            prediction = np.sum((baseline, deviation))
            return prediction
        

    def rating_weighted_advanced(self, user, item, algo, k=None):
        neighbors = self.find_neighbors_of(user, algo, k)
        if len(neighbors) == 0:
            return self.avuserrat[user]
        
        difference = np.array([])
        similarity = np.array([])
        miu = np.mean(self.avbusrat)
        
        for neighbor in neighbors:
            name = neighbor[0]
            score = neighbor[1]
            if item in self.arr[name]:
                neighbor_baseline = np.subtract(np.sum((self.avuserrat[name], self.avbusrat[item])), miu)
                difference = np.append(difference, np.subtract(self.arr[name][item], neighbor_baseline))
                similarity = np.append(similarity, float(score))
        
        if len(difference) == 0:
            return self.avuserrat[user]
        else:
            numerator = np.sum(np.multiply(difference, similarity))
            denominator = np.sum(similarity)
            baseline = np.subtract(np.sum((self.avuserrat[user], self.avbusrat[item])), miu)
            if denominator == 0:
                return baseline
            deviation = np.divide(numerator, denominator)
            prediction = np.sum((baseline, deviation))
            return prediction
        
    def rmsd(self, algo = 'Pearsons', topk = 2):
        sd, count, p= 0, 0, 0
        for person, items in self.arr.iteritems(): #batch.iteritems():
            sim = self.find_neighbors_of(person, algo)
            for other in sim:
                for item, rating in items.items():
                    if item in self.arr[other[0]]:
                        prediction = self.rating_weighted_advanced(person, item, algo, topk)
                        prediction = pd.to_numeric(prediction)
                        truth = pd.to_numeric(rating)
                        difference = np.subtract(prediction,truth)
                        sd += np.power(difference, 2)
                        count+=1
            p+=1
            print(person, p)
        rmsd = np.sqrt(sd / count)
        return rmsd
    
    def bf_rmsd(self, algo = 'Pearson_LinLi', topk = 2):
        has_bff = self.best_friends()
        people = has_bff.as_matrix().flatten()
        narr = self.arr[people]
        sd, count, = 0, 0
        for person, items in narr.iteritems(): #batch.iteritems():
            sim = self.find_neighbors_of(person, algo)
            for other in sim:
                for item, rating in items.items():
                    if item in self.arr[other[0]]:
                        prediction = self.rating_weighted_advanced(person, item, algo, topk)
                        prediction = pd.to_numeric(prediction)
                        truth = pd.to_numeric(rating)
                        difference = np.subtract(prediction,truth)
                        sd += np.power(difference, 2)
                        count+=1
        rmsd = np.sqrt(sd / count)
        return rmsd

    def best_friends(self,k=20):
        bfriends = pd.DataFrame([[]])
        for person, items in self.arr.iteritems():
            sim = self.find_neighbors_of(person, algo= 'Pearson_LinLi')
            bf = sim[0]
            bf = [person, bf[0], bf[1]]
            #print(bf)
            bfriends = bfriends.append(bf)
        bfriends = bfriends.sort_values(by=0, ascending=False)
        return bfriends[0:k]

    def build_train(self, filename = 'train.csv', algo = 'Pearson_LinLi', maxr = 3):
        count, i = 0, 0
        #trainingset = pd.DataFrame([[]])
        output = open(pwd+filename, 'w')
        for user, items in self.arr.iteritems():
            sim = self.find_neighbors_of(user, algo)
            for other in sim:
                #print(other[1],other[0])
                for item in items:
                    if item in self.arr[other[0]]:
                        output.write(str(other[1])+",")
                        output.write(str(self.avbusrat[item])+",")
                        output.write(str(self.avuserrat[user])+",")
                        output.write(str(self.arr[user][item])+"\n") 
                        i += 1
            count+=1
            
    def fit_nn(self, epochs = 100):
        import random
        import tensorflow as tf
        from sklearn.cross_validation import train_test_split as split
        X = pd.read_csv(pwd + 'train.csv', header = None)

        y = X.pop(3)
        X = X.as_matrix()
        y = y.as_matrix()

        trainX, testX, trainY, testY = split(X, y, test_size=0.33, 
                                             random_state=66)

        # Parameters
        learning_rate = 0.001
        batch_size = 30000
        display_step = 1
        training_epochs = epochs
        
        # Network Parameters
        n_hidden_1 = 256 # 1st layer number of features
        n_hidden_2 = 256 # 2nd layer number of features
        n_input = 3
        n_classes = 1
        
        # Store layers weight & bias
        weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

        # tf Graph input
        x = tf.placeholder("float", [None, n_input])
        y = tf.placeholder("float", [None, n_classes])
        
        # Create model
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        
        # Construct model
        #pred = multilayer_perceptron(x, weights, biases)
        
        # Define loss and optimizer
        cost = tf.reduce_mean(tf.squared_difference(out_layer, y))
        
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        
        # Initializing the variables
        init = tf.global_variables_initializer()
        
        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
        
            # Training cycle
            for epoch in range(training_epochs):
                avg_cost = 0.
                num_batches = 5
                # Loop over all batches
                for i in range(num_batches):
                    batch_gen = random.sample(range(0,len(trainX)-1),batch_size)
                    batch_x = trainX[batch_gen]
                    batch_y = trainY[batch_gen].reshape((batch_size, 1))
                    
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                                  y: batch_y})
                    # Compute average loss
                    avg_cost += c / num_batches
                # Display logs per epoch step
                if epoch % display_step == 0:
                    print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            print ("Optimization Finished!")

            
            predictions = out_layer.eval(feed_dict = {x:testX})
            predictions[predictions<1] = 1
            predictions[predictions>5] = 5
            sd, count = 0, 0
            for pred, truth in zip(predictions, testY):
                difference = np.subtract(pred,truth)
                sd += np.power(difference, 2)
                
                count+=1

            rmsd = np.sqrt(sd / count)
            return rmsd
            
