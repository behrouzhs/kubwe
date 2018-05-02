#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <time.h>
#include <omp.h>
#include <cstring>
#include "vptree.h"

#define BUFF_SIZE 16777216 // 16 * 1024 * 1024
#define MAX_FILE_NAME 2048
#define MAX_LINE_LENGTH 131072
#define MAX_WORD_LENGTH 64
#define ALPHA_INIT 0.1

#define DEFAULT_MAX_ITER 100
#define DEFAULT_NO_DIM 100
#define DEFAULT_KERNEL_TYPE 1
#define DEFAULT_KERNEL_PARAM 2
#define DEFAULT_NN_NEGATIVE 2000
#define DEFAULT_UNIT_BALL_CONSTRAINT 1
#define DEFAULT_TAKE_LOG 0
#define DEFAULT_TAKE_SQRT 0
#define DEFAULT_NO_THREAD 4
#define DEFAULT_VERBOSE 1
#define DEFAULT_SAVE 0

//---------- portable timing ----------
#ifdef _WIN32
double time_gettime()
{
	return ((double)((double)clock() / (double)CLOCKS_PER_SEC));
}

double time_duration(double begin)
{
	return (time_gettime() - begin);
}
#else
double time_gettime()
{
	struct timespec now;
	clock_gettime(CLOCK_REALTIME, &now);
	return ((double)((double)now.tv_sec + ((double)now.tv_nsec * 1.0e-9)));
}

double time_duration(double begin)
{
	struct timespec now;
	clock_gettime(CLOCK_REALTIME, &now);
	double d_now = (double)((double)now.tv_sec + ((double)now.tv_nsec * 1.0e-9));
	return (d_now - begin);
}
#endif
//---------- portable timing ----------


typedef struct collocation {
	int row;
	int col;
	double val;
} COLLOC;


COLLOC* read_input_file(char *input_file, long *num_rec)
{
	FILE *fid = fopen(input_file, "rb");
	if (fid == NULL) { printf("Error: could not open input file \"%s\".\r\n", input_file); exit(EXIT_FAILURE); }

	fseek(fid, 0, SEEK_END);
	long file_size = ftell(fid);
	(*num_rec) = file_size / sizeof(COLLOC);
	COLLOC *cooccur = (COLLOC*)malloc(sizeof(COLLOC) * (*num_rec));
	if (cooccur == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }

	rewind(fid);
	if ((long)fread(cooccur, sizeof(COLLOC), *num_rec, fid) == (*num_rec))
	{
		fclose(fid);
		return cooccur;
	}
	else
	{
		fclose(fid);
		exit(EXIT_FAILURE);
	}
}


void write_output_file(char *vocab_file, char *output_file, double *Y, int num_row, int no_dim)
{
	FILE *fv = fopen(vocab_file, "rt");
	if (fv == NULL) { printf("Error: could not open vocab file \"%s\".\r\n", vocab_file); exit(EXIT_FAILURE); }

	FILE *fo = fopen(output_file, "wt");
	if (fo == NULL) { printf("Error: could not open output file for writing \"%s\".\r\n", output_file); exit(EXIT_FAILURE); }

	char fmt[32];
	char *word = (char*)malloc(sizeof(char) * MAX_FILE_NAME);
	if (word == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }
	sprintf(fmt, "%%%ds", MAX_FILE_NAME);

	int i, j, n_ind;
	for (i = 0; i < num_row; i++)
	{
		n_ind = i * no_dim;
		if (fscanf(fv, fmt, word) == 0) { fclose(fv); fclose(fo); exit(EXIT_FAILURE); }

		fprintf(fo, "%s", word);
		for (j = 0; j < no_dim; j++)
			fprintf(fo, " %.6f", Y[n_ind + j]);
		fprintf(fo, "\n");

		if (fscanf(fv, fmt, word) == 0) { fclose(fv); fclose(fo); exit(EXIT_FAILURE); }
	}

	fclose(fv);
	fclose(fo);
	free(word); word = NULL;
}


void copy_colloc(COLLOC *cooccur, long num_rec, int *col, double *val)
{
	long i;
	for (i = 0; i < num_rec; i++)
	{
		col[i] = cooccur[i].col;
		val[i] = cooccur[i].val;
	}
}


/* Assumes 1-based indexing in cooccurrences but changes them to 0-based */
void get_rowcol_make0ind(COLLOC *cooccur, long num_rec, long *num_row, long *num_col)
{
	long i;
	*num_row = -1;
	*num_col = -1;
	for (i = 0; i < num_rec; i++)
	{
		if (cooccur[i].row > *num_row)
			*num_row = cooccur[i].row;
		if (cooccur[i].col > *num_col)
			*num_col = cooccur[i].col;

		cooccur[i].row--;
		cooccur[i].col--;
	}
}

void calc_stats_normalize(COLLOC *cooccur, long num_rec, long num_row, int take_log, int take_sqrt, long *start_index)
{
	long i, last_row = -1;

	if (take_log >= 1)
		for (i = 0; i < num_rec; i++)
			cooccur[i].val = log2(cooccur[i].val);
	else if (take_sqrt >= 1)
		for (i = 0; i < num_rec; i++)
			cooccur[i].val = sqrt(cooccur[i].val);

	double *max_row = (double*)malloc(sizeof(double) * num_row);
	if (max_row == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }

	for (i = 0; i < num_row; i++)
		max_row[i] = -DBL_MAX;

	for (i = 0; i < num_rec; i++)
	{
		if (cooccur[i].val > max_row[cooccur[i].row])
			max_row[cooccur[i].row] = cooccur[i].val;

		if (cooccur[i].row != last_row)
		{
			last_row++;
			while (cooccur[i].row != last_row)
			{
				start_index[last_row] = i;
				last_row++;
			}
			start_index[cooccur[i].row] = i;
		}
	}
	start_index[num_row] = num_rec;

	for (i = 0; i < num_rec; i++)
		cooccur[i].val /= max_row[cooccur[i].row];

	free(max_row); max_row = NULL;
}


int compare_colloc(const void *a, const void *b) {
	int diff;
	if ((diff = ((COLLOC*)a)->row - ((COLLOC*)b)->row) != 0)
		return diff;
	else
		return (((COLLOC*)a)->col - ((COLLOC*)b)->col);
}


void initialize_random(int no_point, int no_dim, double *out_Y)
{
	int i, d, n_temp, n_ind;
	double len;

	for (i = 0; i < no_point; ++i)
	{
		n_ind = i * no_dim;
		len = 0;
		for (d = 0; d < no_dim; ++d)
		{
			n_temp = n_ind + d;
			out_Y[n_temp] = ((double)rand() / (double)RAND_MAX) - 0.5;
			len += out_Y[n_temp] * out_Y[n_temp];
		}
		len = sqrt(len);
		if (len > 0)
			for (d = 0; d < no_dim; ++d)
				out_Y[n_ind + d] /= len;
	}
}


int read_embedding(char *embedding_file, double *Y, int no_row, int no_dim)
{
	FILE *fid = fopen(embedding_file, "rt");
	if (fid == NULL) { printf("Error: could not open embedding file \"%s\".\r\n", embedding_file); return 1; }

	int i, j, idx, nz = 0, nr = 0, nd = 0;
	char fmt[32];
	char *line = (char*)malloc(sizeof(char) * MAX_LINE_LENGTH);
	char *word = (char*)malloc(sizeof(char) * MAX_WORD_LENGTH);
	if (line == NULL || word == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }
	sprintf(fmt, "%%%ds", MAX_WORD_LENGTH);

	while (1)
	{
		if (fscanf(fid, fmt, word) <= 0) { break; }
		nr++;
		if (fgets(line, MAX_LINE_LENGTH, fid) == NULL) { fclose(fid); free(line); line = NULL; return 1; }
	}
	if (nr == 0 || nr != no_row) { fclose(fid); free(line); line = NULL; return 1; }

	idx = 0;
	while (1)
	{
		if (sscanf(&line[idx], fmt, word) <= 0) { break; }
		nd++;
		idx += strlen(word) + 1;
		if (line[idx] == 0 || line[idx] == '\n')
			break;
	}
	if (nd == 0 || nd != no_dim) { fclose(fid); free(line); line = NULL; return 1; }

	rewind(fid);
	for (i = 0; i < no_row; i++)
	{
		if (fscanf(fid, fmt, word) <= 0) { fclose(fid); free(line); line = NULL; return 1; }
		if (fgets(line, MAX_LINE_LENGTH, fid) == NULL) { fclose(fid); free(line); line = NULL; return 1; }
		idx = 0;
		for (j = 0; j < no_dim; j++)
		{
			if (sscanf(&line[idx], fmt, word) <= 0) { fclose(fid); free(line); line = NULL; return 1; }
			Y[i * no_dim + j] = atof(word);
			if (Y[i * no_dim + j] != 0)
				nz++;
			idx += strlen(word) + 1;
			if (idx >= (int)strlen(line)) { fclose(fid); free(line); line = NULL; return 1; }
		}
	}
	if (nz <= (no_row * no_dim / 2)) { fclose(fid); free(line); line = NULL; return 1; }

	fclose(fid);
	free(word); word = NULL;
	free(line); line = NULL;
	return 0;
}


void kube_optimize(COLLOC *cooccur, long *start_index, int *cooccur_col, long num_row, int no_dim, int nn_negative, int kernel_type, int kernel_param, int max_iteration, int is_unit_ball, int no_thread, int verbose, double *Y, char *vocab_file, char *output_file, int save)
{
	srand(time(NULL));
	//nn_negative = (int)sqrt(num_row);
	printf("nn_negative: %d\r\n", nn_negative);
	double exp_alpha_pos, duration_create;
	int tid0_processed_cnt = 0, recreated_cnt = 0, recreate_step = num_row / (no_thread * 10);
	int i, j, k, d, iter;
	int n_temp, n_ind, n_jnd, n_tnd;
	double len, alpha = ALPHA_INIT, alpha_base = ALPHA_INIT, alpha_step = (0.9 * ALPHA_INIT / (double)(5.0 * max_iteration));
	if (kernel_type == 1)  // polynomial kernel
		kernel_param--;

	time_t timer;
	char time_buffer[512];
	struct tm* time_info;

	omp_set_num_threads(no_thread);
#pragma omp parallel default(none) shared(no_thread)
	{
#pragma omp master
		no_thread = omp_get_num_threads();
	}
	MaxHeap** heap = maxheap_create_multi(no_thread, nn_negative);
	int *nn_indices = (int*)malloc(sizeof(int) * nn_negative * no_thread);
	int *cnt_neg_total = (int*)malloc(sizeof(int) * no_thread);
	double *nn_distances = (double*)malloc(sizeof(double) * nn_negative * no_thread);
	double *duration = (double*)malloc(sizeof(double) * no_thread);
	double *similarity = (double*)malloc(sizeof(double) * num_row * no_thread);
	double *att_f = (double*)malloc(sizeof(double) * no_dim * no_thread);
	double *rep_f = (double*)malloc(sizeof(double) * no_dim * no_thread);
	double *gradient = (double*)malloc(sizeof(double) * no_dim * no_thread);
	if (similarity == NULL || att_f == NULL || rep_f == NULL || gradient == NULL || nn_indices == NULL || nn_distances == NULL || duration == NULL || cnt_neg_total == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }

	for (iter = 0; iter < max_iteration; ++iter)
	{
		recreated_cnt = 0;
		tid0_processed_cnt = 0;
		for (i = 0; i < no_thread; ++i)
			cnt_neg_total[i] = 0;
		exp_alpha_pos = (double)((double)(max_iteration - iter) / (double)max_iteration) * 0.16666 + 0.5;
		duration_create = time_gettime();
		std::vector<DataPoint> dp_X(num_row, DataPoint(no_dim, -1, Y));
		for (i = 0; i < num_row; ++i)
			dp_X[i] = DataPoint(no_dim, i, Y + i * no_dim);
		VpTree<DataPoint, euclidean_distance>* tree = new VpTree<DataPoint, euclidean_distance>();
		tree->create(dp_X);
		duration_create = time_duration(duration_create);
		//printf("creation time: %f\r\n", duration_create);

		//double time_temp1 = 0.0;
		//printf("testing heap to check log(k)\r\n");
		//for (j = 10; j < 12000; j *= 10)
		//{
		//	maxheap_reset(heap[0]);
		//	time_temp1 = time_gettime();
		//	for (i = 0; i < 1000; ++i)
		//		tree->search(dp_X[i], j, &nn_indices[0 * nn_negative], &nn_distances[0 * nn_negative], heap[0]);
		//	duration = time_duration(time_temp1);
		//	printf("nn_count: %d, search time: %f\r\n", j, duration);
		//}

		//printf("\r\ntesting vptree to check log(n)\r\n");
		//for (j = 100; j < 120000; j *= 10)
		//{
		//	std::vector<DataPoint> dp_X2(j, DataPoint(no_dim, -1, Y));
		//	for (i = 0; i < j; ++i)
		//		dp_X2[i] = DataPoint(no_dim, i, Y + i * no_dim);
		//	VpTree<DataPoint, euclidean_distance>* tree2 = new VpTree<DataPoint, euclidean_distance>();
		//	tree2->create(dp_X2);

		//	time_temp1 = time_gettime();
		//	for (i = 0; i < 100000; ++i)
		//		tree2->search(dp_X[i], 50, &nn_indices[0 * nn_negative], &nn_distances[0 * nn_negative], heap[0]);
		//	duration = time_duration(time_temp1);
		//	printf("n_word: %d, search time: %f\r\n", j, duration);
		//}
		//exit(0);
		
#pragma omp parallel default(none) private(i, j, k, d, n_ind, n_jnd, n_tnd, len, n_temp, exp_alpha_pos) shared(Y, cooccur, start_index, num_row, no_dim, alpha, similarity, att_f, rep_f, gradient, kernel_type, kernel_param, is_unit_ball, tree, dp_X, heap, nn_indices, nn_distances, nn_negative, duration, cooccur_col, cnt_neg_total, tid0_processed_cnt, recreate_step, recreated_cnt)
	{
		int cnt_neg = 0;
		double min_sim_positive, dd_temp;
		int tid = omp_get_thread_num();
		n_tnd = tid * no_dim;
		duration[tid] = 0;

		#pragma omp for
		for (i = 0; i < num_row; ++i)
		{
			min_sim_positive = DBL_MAX;
			n_ind = i * no_dim;
			// compute attractive force
			for (d = 0; d < no_dim; ++d)
				att_f[n_tnd + d] = 0;
			for (k = start_index[i]; k < start_index[i + 1]; ++k)
			{
				n_temp = cooccur[k].col * no_dim;
				for (d = 0; d < no_dim; ++d)
					att_f[n_tnd + d] += cooccur[k].val * Y[n_temp + d];
			}
			len = 0;
			for (d = 0; d < no_dim; ++d)
				len += att_f[n_tnd + d] * att_f[n_tnd + d];
			len = sqrt(len);
			if (len > 0)
				for (d = 0; d < no_dim; ++d)
					att_f[n_tnd + d] /= len;

			// compute repulsive force
			double time_temp = time_gettime();
			int neg_out = tree->search_notconnected(dp_X[i], nn_negative, cooccur_col, start_index[i], start_index[i + 1], &nn_indices[tid * nn_negative], &nn_distances[tid * nn_negative], heap[tid]);
			duration[tid] += time_duration(time_temp);
			cnt_neg = 0;

			for (d = 0; d < no_dim; ++d)
				rep_f[n_tnd + d] = 0;
			for (j = 0; j < neg_out; ++j)
			{
				n_temp = tid * nn_negative + j;
				++cnt_neg;
				++cnt_neg_total[tid];
				n_jnd = nn_indices[n_temp] * no_dim;
				dd_temp = nn_distances[n_temp];
				for (d = 0; d < kernel_param; ++d)
					nn_distances[n_temp] *= dd_temp;
				for (d = 0; d < no_dim; ++d)
					rep_f[n_tnd + d] += nn_distances[n_temp] * Y[n_jnd + d];
			}
			
			len = 0;
			for (d = 0; d < no_dim; ++d)
				len += rep_f[n_tnd + d] * rep_f[n_tnd + d];
			len = sqrt(len);
			if (len > 0)
				for (d = 0; d < no_dim; ++d)
					rep_f[n_tnd + d] /= len;

			// compute final gradient
			len = 0;
			for (d = 0; d < no_dim; ++d)
			{
				n_temp = n_tnd + d;
				gradient[n_temp] = att_f[n_temp] - rep_f[n_temp];
				len += gradient[n_temp] * gradient[n_temp];
			}
			len = sqrt(len);
			if (len > 0)
				for (d = 0; d < no_dim; ++d)
					gradient[n_tnd + d] /= len;

			// update point
			len = 0;
			for (d = 0; d < no_dim; ++d)
			{
				n_temp = n_ind + d;
				Y[n_temp] = Y[n_temp] + (alpha * gradient[n_tnd + d]);
				len += Y[n_temp] * Y[n_temp];
			}
			len = sqrt(len);
			if (is_unit_ball && len > 0)
				for (d = 0; d < no_dim; ++d)
					Y[n_ind + d] /= len;
		}
	}

	if (verbose > 0 && (iter == 0 || (iter + 1) % verbose == 0))
	{
		time(&timer);
		time_info = localtime(&timer);
		strftime(time_buffer, 128, "%Y-%m-%d - %H:%M:%S", time_info);

		double dur = 0.0, exp_neg_count = 0.0;
		for (i = 0; i < no_thread; ++i)
		{
			dur += duration[i];
			exp_neg_count += (double)cnt_neg_total[i];
		}
		printf("%s, iter: %d, alpha: %f, avg_neg: %f, recreate: %d, vptree search time: %f\r\n", time_buffer, iter + 1, alpha, exp_neg_count / (double)num_row, recreated_cnt, dur);
	}

	if (save > 0 && ((iter + 1) % save == 0))
	{
		sprintf(time_buffer, "%s_iter%d.txt", output_file, (iter + 1));
		write_output_file(vocab_file, time_buffer, Y, num_row, no_dim);
	}

	// update learning rate
	alpha_base -= alpha_step;
	alpha = (alpha_base * 13.0 / ALPHA_INIT) - 6.0;
	alpha = 1.0 / (1.0 + exp(-alpha));
	alpha = (alpha * 0.9 * ALPHA_INIT) + (0.1 * ALPHA_INIT);
	}

	free(similarity); similarity = NULL;
	free(att_f); att_f = NULL;
	free(rep_f); rep_f = NULL;
	free(gradient); gradient = NULL;
}


void kube(char *cooccur_file, char *vocab_file, char *output_file, char *init_file, int no_dim, int nn_negative, int take_log, int take_sqrt, int kernel_type, int kernel_param, int max_iteration, int is_unit_ball, int no_thread, int verbose, int save)
{
	double tm_start, tm_dur;
	long num_rec = 0, num_row = 0, num_col = 0, i = 0;

	printf("INITIALIZING...\r\n");
	if (kernel_type == 1)
		printf("Kernel: polynomial  -  degree: %d\r\n", kernel_param);
	else
		printf("Kernel: tanh  -  parameter: %d\r\n", kernel_param);
	tm_start = time_gettime();
	COLLOC *cooccur = read_input_file(cooccur_file, &num_rec);
	tm_dur = time_duration(tm_start);
	printf("Input file has been read in %f seconds: %ld records\r\n", tm_dur, num_rec);

	get_rowcol_make0ind(cooccur, num_rec, &num_row, &num_col);
	printf("# of rows: %ld, # of columns: %ld, sparsity: %.3f%%\r\n", num_row, num_col, 100.0 * (1.0 - (double)((double)num_rec / (double)(num_row * num_col))));
	tm_start = time_gettime();
	qsort(cooccur, num_rec, sizeof(COLLOC), compare_colloc);
	tm_dur = time_duration(tm_start);
	printf("Cooccurrences sorted in %f seconds.\r\n", tm_dur);

	long *start_index = (long*)malloc(sizeof(long) * (num_row + 1));
	if (start_index == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }
	tm_start = time_gettime();
	calc_stats_normalize(cooccur, num_rec, num_row, take_log, take_sqrt, start_index);
	tm_dur = time_duration(tm_start);
	printf("Statistics normalized by row in %f seconds.\r\n", tm_dur);

	int *cooccur_col = (int*)malloc(sizeof(int) * num_rec);
	if (cooccur_col == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }
	for (i = 0; i < num_rec; ++i)
		cooccur_col[i] = cooccur[i].col;

	double *Y = (double*)malloc(sizeof(double) * no_dim * num_row);
	if (Y == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }

	if (init_file == NULL || read_embedding(init_file, Y, num_row, no_dim))
	{
		printf("Initializing randomly...\r\n");
		initialize_random(num_row, no_dim, Y);
	}
	else
		printf("Embedding initialized using the given input file.\r\n");
	printf("EMBEDDING...\r\n");
	
	tm_start = time_gettime();
	kube_optimize(cooccur, start_index, cooccur_col, num_row, no_dim, nn_negative, kernel_type, kernel_param, max_iteration, is_unit_ball, no_thread, verbose, Y, vocab_file, output_file, save);
	tm_dur = time_duration(tm_start);
	printf("Optimization done in %f seconds.\r\nWriting the embeddings to the output file.\r\n", tm_dur);
	write_output_file(vocab_file, output_file, Y, num_row, no_dim);
	printf("All done!\r\n");

	free(cooccur); cooccur = NULL;
	free(start_index); start_index = NULL;
	free(cooccur_col); cooccur_col = NULL;
	free(Y); Y = NULL;
}


int scmp(char *s1, char *s2) {
	while (*s1 != '\0' && *s1 == *s2) { s1++; s2++; }
	return(*s1 - *s2);
}


int find_arg(char *str, int argc, char **argv) {
	int i;
	for (i = 1; i < argc; i++) {
		if (!scmp(str, argv[i])) {
			if (i == argc - 1) {
				printf("No argument given for %s\n", str);
				exit(EXIT_FAILURE);
			}
			return i;
		}
	}
	return -1;
}


void usage()
{
	exit(EXIT_FAILURE);
}


int main(int argc, char **argv)
{
	int i, no_dim = DEFAULT_NO_DIM, no_thread = DEFAULT_NO_THREAD, kernel_type = DEFAULT_KERNEL_TYPE, kernel_param = DEFAULT_KERNEL_PARAM, verbose = DEFAULT_VERBOSE, save = DEFAULT_SAVE;
	int take_log = DEFAULT_TAKE_LOG, take_sqrt = DEFAULT_TAKE_SQRT, is_unit_ball = DEFAULT_UNIT_BALL_CONSTRAINT, max_iteration = DEFAULT_MAX_ITER, nn_negative = DEFAULT_NN_NEGATIVE;
	char *init_file = NULL;
	char *cooccur_file = (char*)malloc(sizeof(char) * MAX_FILE_NAME);
	char *vocab_file = (char*)malloc(sizeof(char) * MAX_FILE_NAME);
	char *output_file = (char*)malloc(sizeof(char) * MAX_FILE_NAME);
	if (cooccur_file == NULL || vocab_file == NULL || output_file == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }

	if ((i = find_arg((char *)"-i", argc, argv)) > 0 || (i = find_arg((char *)"-input", argc, argv)) > 0)
		strcpy(cooccur_file, argv[i + 1]);
	else
		usage();

	if ((i = find_arg((char *)"-b", argc, argv)) > 0 || (i = find_arg((char *)"-vocab", argc, argv)) > 0)
		strcpy(vocab_file, argv[i + 1]);
	else
		usage();

	if ((i = find_arg((char *)"-o", argc, argv)) > 0 || (i = find_arg((char *)"-output", argc, argv)) > 0)
		strcpy(output_file, argv[i + 1]);
	else
		strcpy(output_file, (char*)"embedding.txt");

	if ((i = find_arg((char *)"-init", argc, argv)) > 0 || (i = find_arg((char *)"-init-file", argc, argv)) > 0)
	{
		init_file = (char*)malloc(sizeof(char) * MAX_FILE_NAME);
		if (init_file == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }
		strcpy(init_file, argv[i + 1]);
	}

	if ((i = find_arg((char *)"-d", argc, argv)) > 0 || (i = find_arg((char *)"-dim", argc, argv)) > 0)
		no_dim = atoi(argv[i + 1]);
	if ((i = find_arg((char *)"-n", argc, argv)) > 0 || (i = find_arg((char *)"-neg", argc, argv)) > 0)
		nn_negative = atoi(argv[i + 1]);
	if ((i = find_arg((char *)"-t", argc, argv)) > 0 || (i = find_arg((char *)"-thread", argc, argv)) > 0)
		no_thread = atoi(argv[i + 1]);
	if ((i = find_arg((char *)"-k", argc, argv)) > 0 || (i = find_arg((char *)"-kernel", argc, argv)) > 0)
	{
		if (!scmp(argv[i + 1], (char*)"poly"))
			kernel_type = 1;
		else  // tanh
			kernel_type = 2;
	}
	if ((i = find_arg((char *)"-kp", argc, argv)) > 0 || (i = find_arg((char *)"-kparam", argc, argv)) > 0)
		kernel_param = atoi(argv[i + 1]);
	if ((i = find_arg((char *)"-v", argc, argv)) > 0 || (i = find_arg((char *)"-verbose", argc, argv)) > 0)
		verbose = atoi(argv[i + 1]);
	if ((i = find_arg((char *)"-sv", argc, argv)) > 0 || (i = find_arg((char *)"-save", argc, argv)) > 0)
		save = atoi(argv[i + 1]);
	if ((i = find_arg((char *)"-it", argc, argv)) > 0 || (i = find_arg((char *)"-iter", argc, argv)) > 0)
		max_iteration = atoi(argv[i + 1]);
	if ((i = find_arg((char *)"-l", argc, argv)) > 0 || (i = find_arg((char *)"-log", argc, argv)) > 0)
		take_log = 1;
	if ((i = find_arg((char *)"-sq", argc, argv)) > 0 || (i = find_arg((char *)"-sqrt", argc, argv)) > 0)
		take_sqrt = 1;
	if ((i = find_arg((char *)"-ul", argc, argv)) > 0 || (i = find_arg((char *)"-norm", argc, argv)) > 0 || (i = find_arg((char *)"-normalize", argc, argv)) > 0)
		is_unit_ball = atoi(argv[i + 1]);

	kube(cooccur_file, vocab_file, output_file, init_file, no_dim, nn_negative, take_log, take_sqrt, kernel_type, kernel_param, max_iteration, is_unit_ball, no_thread, verbose, save);

	free(cooccur_file); cooccur_file = NULL;
	free(vocab_file); vocab_file = NULL;
	free(output_file); output_file = NULL;
	return EXIT_SUCCESS;
}
