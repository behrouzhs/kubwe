/*
 *
 * Copyright (c) 2014, Laurens van der Maaten (Delft University of Technology)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    This product includes software developed by the Delft University of Technology.
 * 4. Neither the name of the Delft University of Technology nor the names of 
 *    its contributors may be used to endorse or promote products derived from 
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY LAURENS VAN DER MAATEN ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO 
 * EVENT SHALL LAURENS VAN DER MAATEN BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR 
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING 
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY 
 * OF SUCH DAMAGE.
 *
 */


/* This code was adopted with minor modifications from Steve Hanov's great tutorial at http://stevehanov.ca/blog/index.php?id=130 */

#include <stdlib.h>
#include <algorithm>
#include <vector>
#include <stdio.h>
#include <queue>
#include <limits>
#include <cmath>


#ifndef VPTREE_H
#define VPTREE_H


class DataPoint
{
    int _ind;

public:
    double* _x;
    int _D;
    DataPoint() {
        _D = 1;
        _ind = -1;
        _x = NULL;
    }
    DataPoint(int D, int ind, double* x) {
        _D = D;
        _ind = ind;
        _x = (double*) malloc(_D * sizeof(double));
        for(int d = 0; d < _D; d++) _x[d] = x[d];
    }
    DataPoint(const DataPoint& other) {                     // this makes a deep copy -- should not free anything
        if(this != &other) {
            _D = other.dimensionality();
            _ind = other.index();
            _x = (double*) malloc(_D * sizeof(double));      
            for(int d = 0; d < _D; d++) _x[d] = other.x(d);
        }
    }
    ~DataPoint() { if(_x != NULL) free(_x); }
    DataPoint& operator= (const DataPoint& other) {         // asignment should free old object
        if(this != &other) {
            if(_x != NULL) free(_x);
            _D = other.dimensionality();
            _ind = other.index();
            _x = (double*) malloc(_D * sizeof(double));
            for(int d = 0; d < _D; d++) _x[d] = other.x(d);
        }
        return *this;
    }
    int index() const { return _ind; }
    int dimensionality() const { return _D; }
    double x(int d) const { return _x[d]; }
};


double euclidean_distance(const DataPoint &t1, const DataPoint &t2) {
    double dd = 0.0;
    double* x1 = t1._x;
    double* x2 = t2._x;
	int i;
	for (i = 0; i < t1._D; ++i)
		dd += x1[i] * x2[i];
	return 1.0 - dd;

    //double diff;
    //for(int d = 0; d < t1._D; d++) {
    //    diff = (x1[d] - x2[d]);
    //    dd += diff * diff;
    //}
    //return sqrt(dd);
}


// An item on the intermediate result queue
struct HeapItem {
	HeapItem(int index, double dist) :
		index(index), dist(dist) {}
	int index;
	double dist;
	bool operator<(const HeapItem& o) const {
		return dist < o.dist;
	}
};


typedef struct _MaxHeapItem
{
	double dist;
	int index;
}MaxHeapItem;


typedef struct _MaxHeap
{
	int no_items;
	MaxHeapItem **elements;
}MaxHeap;


MaxHeap* maxheap_create(int capacity)
{
	// we don't use index 0 for easier indexing
	int i;
	++capacity;
	MaxHeap *heap = (MaxHeap*)malloc(sizeof(MaxHeap));
	if (heap == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }
	heap->elements = (MaxHeapItem**)malloc(sizeof(MaxHeapItem*) * capacity);
	if (heap->elements == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }
	for (i = 0; i < capacity; ++i)
	{
		heap->elements[i] = (MaxHeapItem*)malloc(sizeof(MaxHeapItem));
		if (heap->elements[i] == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }
	}

	heap->elements[0]->dist = DBL_MAX;
	heap->no_items = 0;
	return heap;
}


MaxHeap** maxheap_create_multi(int no_heap, int capacity)
{
	int i;
	MaxHeap **heap = (MaxHeap**)malloc(sizeof(MaxHeap*) * no_heap);
	if (heap == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }
	for (i = 0; i < no_heap; ++i)
		heap[i] = maxheap_create(capacity);

	return heap;
}


void maxheap_reset(MaxHeap *heap)
{
	heap->no_items = 0;
}


void maxheap_reset_multi(MaxHeap **heap, int no_heap)
{
	int i;
	for (i = 0; i < no_heap; ++i)
		heap[i]->no_items = 0;
}


void maxheap_destroy(MaxHeap *heap)
{
	free(heap->elements); heap->elements = NULL;
	free(heap); heap = NULL;
}


void maxheap_destroy_multi(MaxHeap **heap, int no_heap)
{
	int i;
	for (i = 0; i < no_heap; ++i)
	{
		free(heap[i]->elements); heap[i]->elements = NULL;
		free(heap[i]); heap[i] = NULL;
	}
	free(heap); heap = NULL;
}


void maxheap_push(MaxHeap *heap, double dist, int index)
{
	++(heap->no_items);
	int now = heap->no_items;
	int parent = now / 2;
	while (heap->elements[parent]->dist < dist)
	{
		*heap->elements[now] = *heap->elements[parent];
		now = parent;
		parent = now / 2;
	}
	heap->elements[now]->dist = dist;
	heap->elements[now]->index = index;
}


void maxheap_pop(MaxHeap *heap, double *dist, int *index)
{
	double lastElement;
	int child, now, lastIndex;
	*dist = heap->elements[1]->dist;
	*index = heap->elements[1]->index;
	lastElement = heap->elements[heap->no_items]->dist;
	lastIndex = heap->elements[heap->no_items]->index;
	--(heap->no_items);

	for (now = 1, child = 2; child <= heap->no_items; now = child, child = now * 2)
	{
		if (child < heap->no_items && heap->elements[child + 1]->dist > heap->elements[child]->dist)
			child++;
		if (lastElement < heap->elements[child]->dist)
			*heap->elements[now] = *heap->elements[child];
		else
			break;
	}
	heap->elements[now]->dist = lastElement;
	heap->elements[now]->index = lastIndex;
}


void maxheap_pop_discard(MaxHeap *heap)
{
	double lastElement;
	int child, now, lastIndex;
	lastElement = heap->elements[heap->no_items]->dist;
	lastIndex = heap->elements[heap->no_items]->index;
	--(heap->no_items);

	for (now = 1, child = 2; child <= heap->no_items; now = child, child = now * 2)
	{
		if (child < heap->no_items && heap->elements[child + 1]->dist > heap->elements[child]->dist)
			child++;
		if (lastElement < heap->elements[child]->dist)
			*heap->elements[now] = *heap->elements[child];
		else
			break;
	}
	heap->elements[now]->dist = lastElement;
	heap->elements[now]->index = lastIndex;
}


void maxheap_pop_push(MaxHeap *heap, double dist, int index)
{
	int child, now;
	for (now = 1, child = 2; child <= heap->no_items; now = child, child = now * 2)
	{
		if (child < heap->no_items && heap->elements[child + 1]->dist > heap->elements[child]->dist)
			child++;
		if (dist < heap->elements[child]->dist)
			*heap->elements[now] = *heap->elements[child];
		else
			break;
	}
	heap->elements[now]->dist = dist;
	heap->elements[now]->index = index;
}


int exists_binarysearch(int *cooccur_col, int target, int start, int end)
{
	--end;
	int mid;
	while (start <= end)
	{
		mid = (start + end) / 2;
		if (target > cooccur_col[mid])
			start = mid + 1;
		else if (target < cooccur_col[mid])
			end = mid - 1;
		else
			return 1;
	}
	return 0;
}


//double test_approximate_vptree_accuracy(double *Y, long num_row, int no_dim, int no_thread)
//{
//	int nn_negative = 1000, i;
//	std::vector<DataPoint> dp_X(num_row, DataPoint(no_dim, -1, Y));
//	for (i = 0; i < num_row; ++i)
//		dp_X[i] = DataPoint(no_dim, i, &Y[i * no_dim]);
//	VpTree<DataPoint, euclidean_distance>* tree = new VpTree<DataPoint, euclidean_distance>();
//	tree->create(dp_X);
//
//	int *nn_indices = (int*)malloc(sizeof(int) * nn_negative * no_thread);
//	double *nn_distances = (double*)malloc(sizeof(double) * nn_negative * no_thread);
//}


struct Node
{
	int index;              // index of point in node
	double threshold;       // radius(?)
	Node* left;             // points closer by than threshold
	Node* right;            // points farther away than threshold

	Node() :
		index(0), threshold(0.), left(0), right(0) {}

	~Node() {               // destructor
		delete left;
		delete right;
	}

	Node* Copy() {
		Node* node = (Node*)malloc(sizeof(Node));
		node->index = index;
		node->threshold = threshold;
		if (left != NULL)
			node->left = left->Copy();
		else
			node->left = NULL;
		if (right != NULL)
			node->right = right->Copy();
		else
			node->right = NULL;
		return node;
	}
};


template<typename T, double (*distance)( const T&, const T& )>
class VpTree
{
public:
    
    // Default constructor
    VpTree() : _root(0) {}
    
    // Destructor
    ~VpTree() {
        delete _root;
    }

	void free_resources()
	{
		if (_root != NULL)
			delete _root;
		//int i;
		//for (i = 0; i < _items.size(); ++i)
		//	((DataPoint)_items[i]).~DataPoint();
		_items.~vector();
	}

    // Function to create a new VpTree from data
    void create(const std::vector<T>& items) {
		if (_root != NULL)
			delete _root;
        _items = items;
        _root = buildFromPoints(0, items.size());
    }

	VpTree* Copy() {
		VpTree<DataPoint, euclidean_distance>* tree = new VpTree<DataPoint, euclidean_distance>();
		tree->_root = _root->Copy();
		tree->_items = _items;
		return tree;
	}
    
    // Function that uses the tree to find the k nearest neighbors of target
    void search(const T& target, int k, int* nn_indices, double* nn_distances, MaxHeap* heap)
    {
		maxheap_reset(heap);
		double tau = DBL_MAX;
		// Perform the search
        search(_root, &tau, target, k, heap);

		int i;
		for (i = 0; i < k; ++i)
			maxheap_pop(heap, &nn_distances[i], &nn_indices[i]);
		for (i = 0; i < k; ++i)
			nn_distances[i] = 1 - nn_distances[i];
        
        // Results are in reverse order
        //std::reverse(results->begin(), results->end());
        //std::reverse(distances->begin(), distances->end());
    }


	int search_notconnected(const T& target, int k, int *cooccur_col, int start, int end, int* nn_indices, double* nn_distances, MaxHeap* heap)
	{
		maxheap_reset(heap);
		double tau = DBL_MAX;
		double min_dist = 1.0 - (3.0 / sqrt(target.dimensionality()));
		//min_dist = 0.0;
		search_recursive_notconnected(_root, &tau, min_dist, target, cooccur_col, start, end, k, heap);
		int k_out = heap->no_items;

		int i;
		for (i = 0; i < k_out; ++i)
			maxheap_pop(heap, &nn_distances[i], &nn_indices[i]);
		for (i = 0; i < k_out; ++i)
			nn_distances[i] = 1 - nn_distances[i];
		return k_out;
	}
    
private:
    std::vector<T> _items;
    
    // Single node of a VP tree (has a point and radius; left children are closer to point than the radius)
    Node* _root;
    
    
    // Distance comparator for use in std::nth_element
    struct DistanceComparator
    {
        const T& item;
        DistanceComparator(const T& item) : item(item) {}
        bool operator()(const T& a, const T& b) {
            return distance(item, a) < distance(item, b);
        }
    };
    
    // Function that (recursively) fills the tree
    Node* buildFromPoints( int lower, int upper )
    {
        if (upper == lower) {     // indicates that we're done here!
            return NULL;
        }
        
        // Lower index is center of current node
        Node* node = new Node();
        node->index = lower;
        
        if (upper - lower > 1) {      // if we did not arrive at leaf yet
            
            // Choose an arbitrary point and move it to the start
            int i = (int) ((double)rand() / RAND_MAX * (upper - lower - 1)) + lower;
            std::swap(_items[lower], _items[i]);
            
            // Partition around the median distance
            int median = (upper + lower) / 2;
            std::nth_element(_items.begin() + lower + 1,
                             _items.begin() + median,
                             _items.begin() + upper,
                             DistanceComparator(_items[lower]));
            
            // Threshold of the new node will be the distance to the median
            node->threshold = distance(_items[lower], _items[median]);
            
            // Recursively build tree
            node->index = lower;
            node->left = buildFromPoints(lower + 1, median);
            node->right = buildFromPoints(median, upper);
        }
        
        // Return result
        return node;
    }
    
    // Helper function that searches the tree    
    void search(const Node* node, double *tau, const T& target, int k, MaxHeap *heap)
    {
        if(node == NULL) return;     // indicates that we're done here
        
        // Compute distance between target and current node
        //double dist = distance(_items[node->index], target);
		double dist = 0.0;
		int i;
		for (i = 0; i < target.dimensionality(); ++i)
			dist += _items[node->index]._x[i] * target._x[i];
		dist = 1 - dist;

        // If current node within radius tau
		if (heap->no_items < k)
			maxheap_push(heap, dist, _items[node->index].index());
		else if (dist < (*tau))
		{
			maxheap_pop_push(heap, dist, _items[node->index].index());
			(*tau) = heap->elements[1]->dist;
		}

        
        // Return if we arrived at a leaf
        if(node->left == NULL && node->right == NULL) {
            return;
        }
        
        // If the target lies within the radius of ball
        if(dist < node->threshold) {
            if(dist - (*tau) <= node->threshold) {         // if there can still be neighbors inside the ball, recursively search left child first
                search(node->left, tau, target, k, heap);
            }
            
            if(dist + (*tau) >= node->threshold) {         // if there can still be neighbors outside the ball, recursively search right child
                search(node->right, tau, target, k, heap);
            }
        
        // If the target lies outsize the radius of the ball
        } else {
            if(dist + (*tau) >= node->threshold) {         // if there can still be neighbors outside the ball, recursively search right child first
                search(node->right, tau, target, k, heap);
            }
            
            if (dist - (*tau) <= node->threshold) {         // if there can still be neighbors inside the ball, recursively search left child
                search(node->left, tau, target, k, heap);
            }
        }
    }


	void search_recursive_notconnected(const Node* node, double *tau, double min_dist, const T& target, int *cooccur_col, int start, int end, int k, MaxHeap *heap)
	{
		if (node == NULL) return;

		double dist = 0.0;
		int i;
		for (i = 0; i < target.dimensionality(); ++i)
			dist += _items[node->index]._x[i] * target._x[i];
		dist = 1.0 - dist;

		if (target.index() != _items[node->index].index() && exists_binarysearch(cooccur_col, _items[node->index].index(), start, end) == 0)
		{
			if (heap->no_items < k)
			{
				maxheap_push(heap, dist, _items[node->index].index());
				if (heap->no_items == k)
					(*tau) = heap->elements[1]->dist;
			}
			else if (dist < (*tau))
			{
				maxheap_pop_push(heap, dist, _items[node->index].index());
				(*tau) = heap->elements[1]->dist;
			}
		}

		if (node->left == NULL && node->right == NULL)
			return;

		if (dist < node->threshold) {
			search_recursive_notconnected(node->left, tau, min_dist, target, cooccur_col, start, end, k, heap);
			if (dist + (*tau) - min_dist >= node->threshold) {
				search_recursive_notconnected(node->right, tau, min_dist, target, cooccur_col, start, end, k, heap);
			}
		}
		else {
			search_recursive_notconnected(node->right, tau, min_dist, target, cooccur_col, start, end, k, heap);
			if (dist - (*tau) + min_dist <= node->threshold) {
				search_recursive_notconnected(node->left, tau, min_dist, target, cooccur_col, start, end, k, heap);
			}
		}
	}
};

#endif
