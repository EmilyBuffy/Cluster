"""K-Means++ class made for multiple parameters. Returns k initial seeds to be used in Equal_Cluster_K_Means"""
import numpy as np
import random
import math

class K_Means_Plus_Plus:

    """Input is a 2D list of n-dimensional points"""
    def __init__(self, points_list, k): #假设传入50个n维数据(50*n)
        self.centroid_count = 0 #种子点计数
        self.point_count = len(points_list)
        self.cluster_count = k
        self.points_list = list(points_list)
        self.initialize_random_centroid()
        self.initialize_other_centroids()

    """Picks a random point to serve as the first centroid"""
    def initialize_random_centroid(self):
        self.centroid_list = []
        index = random.randint(0, len(self.points_list)-1)#在下标0-49任意选一个

        self.centroid_list.append(self.remove_point(index))#传入
        self.centroid_count = 1 #种子点计数

    """Removes point associated with given index so it cannot be picked as a future centroid.
    Returns list containing coordinates of newly removed centroid"""
    def remove_point(self, index):
        new_centroid = self.points_list[index] #注意，list 是2D,返回的是该index所在行向量
        del self.points_list[index] #将已经选为中心点的点从现有集合中除去（这里的集合是存放在列表中）

        return new_centroid # 返回第一个种子点

    """Finds the other k-1 centroids from the remaining lists of points"""
    def initialize_other_centroids(self): 
        while not self.is_finished(): #判断是否找到足够的种子点
            distances = self.find_smallest_distances() # 传入distances是具体距离数值，还未归一化
            chosen_index = self.choose_weighted(distances)#返回一个下标
            self.centroid_list.append(self.remove_point(chosen_index))#将选中的新种子点加入centroid_list
            self.centroid_count += 1


    """Calculates distance from each point to its nearest cluster center. Then chooses new
    center based on the weighted probability of these distances"""
    def find_smallest_distances(self):
        distance_list = [] #存放所有min_distance的1D列表

        for point in self.points_list: #逐个点遍历计算其距离哪个种子点最近，
            distance_list.append(self.find_nearest_centroid(point)) # 每次返回一个min_distance，
#循环结束后，返回initialize_other_centroids(self): 
        return distance_list

    """Finds centroid nearest to the given point, and returns its distance"""
    def find_nearest_centroid(self, point):
        min_distance = math.inf  #初始化为一个标量

        for values in self.centroid_list: #目前只有一个中心点 
            distance = self.euclidean_distance(values, point)#self.euclidean_distance 传回的是一个标量值
             #基于上一个find_smallest_distances，不断有点point传入，计算的值都在重新赋值给distance
            if distance < min_distance: # min_distance 只记录最小值
                min_distance = distance

        return min_distance
    
     """computes N-d euclidean distance between two points represented as lists:
     (x1, x2, ..., xn) and (y1, y2, ..., yn)"""
    def euclidean_distance(self, point1, point2):
        point1 = np.asarray(point1)
        point2 = np.asarray(point2)

        return np.linalg.norm(point2-point1) #向量的范数运算，axis=None，默认为欧几里得距离（标量）

    """Chooses an index based on weighted probability"""
    def choose_weighted(self, distance_list):
        distance_list = [x**2 for x in distance_list] #去根号
        weighted_list = self.weight_values(distance_list)
        indices = [i for i in range(len(distance_list))] # 生成1D-array，
        return np.random.choice(indices, p = weighted_list) # 选到概率最大的下标
#为什么要选择离种子点最近的距离记录到distance_list就能保证距离所有的已选择的中心点是最远的的点？？？？
    """Weights values from [0,1]"""
    def weight_values(self, list):
        sum = np.sum(list) 
        return [x/sum for x in list] #返回一个列表，里面的元素分布在[0,1]之间



    """Checks to see if final condition has been satisfied (when K centroids have been created)"""
    def is_finished(self):
        outcome = False
        if self.centroid_count == self.cluster_count:
            outcome = True

        return outcome

    """Returns final centroid values"""
    def final_centroids(self):
        return self.centroid_list

