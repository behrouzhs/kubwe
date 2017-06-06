#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <time.h>
#include <omp.h>

#define BUFF_SIZE 16777216 // 16 * 1024 * 1024
#define MAX_FILE_NAME 2048
#define MAX_LINE_LENGTH 131072
#define MAX_WORD_LENGTH 64
#define ALPHA_INIT 0.1
#define MAX_ITER 50

#define DEFAULT_NO_DIM 100
#define DEFAULT_KERNEL_DEGREE 1
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
	if (fread(cooccur, sizeof(COLLOC), *num_rec, fid) == (*num_rec))
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
			if (idx >= strlen(line)) { fclose(fid); free(line); line = NULL; return 1; }
		}
	}
	if (nz <= (no_row * no_dim / 2)) { fclose(fid); free(line); line = NULL; return 1; }

	fclose(fid);
	free(word); word = NULL;
	free(line); line = NULL;
	return 0;
}


void kube_optimize(COLLOC *cooccur, long *start_index, long num_row, int no_dim, int kernel_poly_degree, int is_unit_ball, int no_thread, int verbose, double *Y, char *vocab_file, char *output_file, int save)
{
	int i, j, k, d, iter;
	int n_temp, n_ind, n_jnd, n_tnp, n_tnd;
	double len, d_temp, alpha = ALPHA_INIT, alpha_base = ALPHA_INIT, alpha_step = (0.9 * ALPHA_INIT / (double)(5.0 * MAX_ITER));
	kernel_poly_degree--;

	time_t timer;
	char time_buffer[512];
	struct tm* time_info;

	omp_set_num_threads(no_thread);
#pragma omp parallel default(none) shared(no_thread)
	{
#pragma omp master
		no_thread = omp_get_num_threads();
	}
	double *similarity = (double*)malloc(sizeof(double) * num_row * no_thread);
	double *att_f = (double*)malloc(sizeof(double) * no_dim * no_thread);
	double *rep_f = (double*)malloc(sizeof(double) * no_dim * no_thread);
	double *gradient = (double*)malloc(sizeof(double) * no_dim * no_thread);
	if (similarity == NULL || att_f == NULL || rep_f == NULL || gradient == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }

	initialize_random(num_row, no_dim, Y);
	for (iter = 0; iter < MAX_ITER; ++iter)
	{
#pragma omp parallel default(none) private(i, j, k, d, n_ind, n_jnd, n_tnd, n_tnp, len, n_temp, d_temp) shared(Y, cooccur, start_index, num_row, no_dim, alpha, similarity, att_f, rep_f, gradient, kernel_poly_degree, is_unit_ball)
	{
		int tid = omp_get_thread_num();
		n_tnd = tid * no_dim;
		n_tnp = tid * num_row;

		#pragma omp for
		for (i = 0; i < num_row; ++i)
		{
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
			for (j = 0; j < num_row; ++j)
			{
				n_temp = n_tnp + j;
				n_jnd = j * no_dim;
				similarity[n_temp] = 0;
				for (d = 0; d < no_dim; ++d)
					similarity[n_temp] += Y[n_jnd + d] * Y[n_ind + d];
				similarity[n_temp] = (similarity[n_temp] + 1.0) / 2.0;

				// poynomial kernel
				d_temp = similarity[n_temp];
				for (d = 0; d < kernel_poly_degree; d++)
					similarity[n_temp] *= d_temp;
			}

			similarity[n_tnp + i] = 0;
			for (k = start_index[i]; k < start_index[i + 1]; ++k)
				similarity[n_tnp + cooccur[k].col] = 0;

			for (d = 0; d < no_dim; ++d)
				rep_f[n_tnd + d] = 0;
			for (j = 0; j < num_row; ++j)
			{
				n_temp = n_tnp + j;
				n_jnd = j * no_dim;
				for (d = 0; d < no_dim; ++d)
					rep_f[n_tnd + d] += similarity[n_temp] * Y[n_jnd + d];
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

		printf("%s, iter: %d\r\n", time_buffer, iter + 1);
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


void kube(char *cooccur_file, char *vocab_file, char *output_file, char *init_file, int no_dim, int take_log, int take_sqrt, int kernel_poly_degree, int is_unit_ball, int no_thread, int verbose, int save)
{
	double tm_start, tm_dur;
	long num_rec = 0, num_row = 0, num_col = 0;

	printf("INITIALIZING...\r\n");
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
	kube_optimize(cooccur, start_index, num_row, no_dim, kernel_poly_degree, is_unit_ball, no_thread, verbose, Y, vocab_file, output_file, save);
	tm_dur = time_duration(tm_start);
	printf("Optimization done in %f seconds.\r\nWriting the embeddings to the output file.\r\n", tm_dur);
	write_output_file(vocab_file, output_file, Y, num_row, no_dim);
	printf("All done!\r\n");

	free(cooccur); cooccur = NULL;
	free(start_index); start_index = NULL;
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
	int i, no_dim = DEFAULT_NO_DIM, no_thread = DEFAULT_NO_THREAD, kernel_poly_degree = DEFAULT_KERNEL_DEGREE, verbose = DEFAULT_VERBOSE, save = DEFAULT_SAVE;
	int take_log = DEFAULT_TAKE_LOG, take_sqrt = DEFAULT_TAKE_SQRT, is_unit_ball = DEFAULT_UNIT_BALL_CONSTRAINT;
	char *init_file = NULL;
	char *cooccur_file = (char*)malloc(sizeof(char) * MAX_FILE_NAME);
	char *vocab_file = (char*)malloc(sizeof(char) * MAX_FILE_NAME);
	char *output_file = (char*)malloc(sizeof(char) * MAX_FILE_NAME);
	if (cooccur_file == NULL || vocab_file == NULL || output_file == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }

	if ((i = find_arg((char *)"-i", argc, argv)) > 0 || (i = find_arg((char *)"-input", argc, argv)) > 0)
		strcpy(cooccur_file, argv[i + 1]);
	else
		usage();

	if ((i = find_arg((char *)"-vo", argc, argv)) > 0 || (i = find_arg((char *)"-vocab", argc, argv)) > 0)
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
	if ((i = find_arg((char *)"-t", argc, argv)) > 0 || (i = find_arg((char *)"-thread", argc, argv)) > 0)
		no_thread = atoi(argv[i + 1]);
	if ((i = find_arg((char *)"-k", argc, argv)) > 0 || (i = find_arg((char *)"-kernel", argc, argv)) > 0)
		kernel_poly_degree = atoi(argv[i + 1]);
	if ((i = find_arg((char *)"-v", argc, argv)) > 0 || (i = find_arg((char *)"-verbose", argc, argv)) > 0)
		verbose = atoi(argv[i + 1]);
	if ((i = find_arg((char *)"-sv", argc, argv)) > 0 || (i = find_arg((char *)"-save", argc, argv)) > 0)
		save = atoi(argv[i + 1]);
	if ((i = find_arg((char *)"-l", argc, argv)) > 0 || (i = find_arg((char *)"-log", argc, argv)) > 0)
		take_log = 1;
	if ((i = find_arg((char *)"-sq", argc, argv)) > 0 || (i = find_arg((char *)"-sqrt", argc, argv)) > 0)
		take_sqrt = 1;
	if ((i = find_arg((char *)"-ul", argc, argv)) > 0 || (i = find_arg((char *)"-norm", argc, argv)) > 0 || (i = find_arg((char *)"-normalize", argc, argv)) > 0)
		is_unit_ball = atoi(argv[i + 1]);

	kube(cooccur_file, vocab_file, output_file, init_file, no_dim, take_log, take_sqrt, kernel_poly_degree, is_unit_ball, no_thread, verbose, save);

	free(cooccur_file); cooccur_file = NULL;
	free(vocab_file); vocab_file = NULL;
	free(output_file); output_file = NULL;
	return EXIT_SUCCESS;
}
