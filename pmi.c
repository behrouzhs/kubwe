#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>

#define BUFF_SIZE 16777216 // 16 * 1024 * 1024
#define MAX_FILE_NAME 2048

#define DEFAULT_WRITE 1
#define DEFAULT_SHIFT 0.0
#define DEFAULT_SMOOTH 0.0
#define DEFAULT_CONTEXT_SMOOTH 0.0
#define DEFAULT_SMOOTH_TYPE 2
#define DEFAULT_CUTP -DBL_MAX
#define DEFAULT_CUTN -DBL_MAX

typedef struct collocation {
	int row;
	int col;
	double val;
} COLLOC;

typedef struct collocation_pmi {
	int row;
	int col;
	double val;
	double pmi;
} COLLOC_P;

typedef struct collocation_npmi {
	int row;
	int col;
	double val;
	double pmi;
	double npmi;
} COLLOC_NP;

void scan_rowcol(char *path, int *nrow, int *ncol, int *nrec)
{
	COLLOC *buffer = (COLLOC*)malloc(sizeof(COLLOC) * BUFF_SIZE);
	if (buffer == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }

	int i, n;
	*nrow = -1;
	*ncol = -1;
	*nrec = 0;
	FILE *fid = fopen(path, "rb");
	if (fid == NULL) { printf("Error: could not open input file \"%s\".\r\n", path); exit(EXIT_FAILURE); }
	do
	{
		n = fread(buffer, sizeof(COLLOC), BUFF_SIZE, fid);
		*nrec += n;
		for (i = 0; i < n; i++)
		{
			if (buffer[i].row > *nrow)
				*nrow = buffer[i].row;
			if (buffer[i].col > *ncol)
				*ncol = buffer[i].col;
		}
	} while (n == BUFF_SIZE);
	fclose(fid);
	free(buffer); buffer = NULL;

	(*nrow)++; // works for both 0-based and 1-based indexing
	(*ncol)++; // works for both 0-based and 1-based indexing
}

void calc_stats(char *path, double *sum_row, int nrow, double *sum_col, int ncol, double *sumsum, double smooth, int smooth_type, double context_dist_smooth)
{
	int i, n;
	COLLOC *buffer = (COLLOC*)malloc(sizeof(COLLOC) * BUFF_SIZE);
	if (buffer == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }

	FILE *fid = fopen(path, "rb");
	if (fid == NULL) { printf("Error: could not open input file \"%s\".\r\n", path); exit(EXIT_FAILURE); }

	*sumsum = 0;
	for (i = 0; i < nrow; i++)
		sum_row[i] = 0.0;
	for (i = 0; i < ncol; i++)
		sum_col[i] = 0.0;

	do
	{
		n = fread(buffer, sizeof(COLLOC), BUFF_SIZE, fid);
		for (i = 0; i < n; i++)
		{
			if (smooth_type >= 2)
			{
				sum_row[buffer[i].row] += (buffer[i].val + smooth);
				sum_col[buffer[i].col] += (buffer[i].val + smooth);
				*sumsum += (buffer[i].val + smooth);
			}
			else
			{
				sum_row[buffer[i].row] += buffer[i].val;
				sum_col[buffer[i].col] += buffer[i].val;
				*sumsum += buffer[i].val;
			}
		}
	} while (n == BUFF_SIZE);
	fclose(fid);
	free(buffer); buffer = NULL;

	if (smooth_type <= 1)
	{
		*sumsum += ((double)nrow * (double)ncol * smooth);
		for (i = 0; i < nrow; i++)
			sum_row[i] += ((double)ncol * smooth);
		for (i = 0; i < ncol; i++)
			sum_col[i] = ((double)nrow * smooth);
	}

	double temp;
	for (i = 0, temp = 0.0; i < nrow; i++)
		temp += sum_row[i];
	for (i = 0; i < nrow; i++)
		sum_row[i] /= temp;

	for (i = 0, temp = 0.0; i < ncol; i++)
	{
		if (context_dist_smooth > DBL_MIN)
			sum_col[i] = pow(sum_col[i], context_dist_smooth);
		temp += sum_col[i];
	}
	for (i = 0; i < ncol; i++)
		sum_col[i] /= temp;
}

void copy_0_np(COLLOC *source, COLLOC_NP *dest)
{
	dest->row = source->row;
	dest->col = source->col;
	dest->val = source->val;
}

void copy_np_0(COLLOC_NP *source, COLLOC *dest, double shift, int write_type)
{
	dest->row = source->row;
	dest->col = source->col;
	if (write_type == 0)
		dest->val = source->val;
	else if (write_type == 1)
		dest->val = source->pmi + shift;
	else
		dest->val = source->npmi + shift;
}

void copy_np_1(COLLOC_NP *source, COLLOC_P *dest, double shift, int write_type)
{
	dest->row = source->row;
	dest->col = source->col;
	if (write_type == 3)
	{
		dest->val = source->val;
		dest->pmi = source->pmi + shift;
	}
	else if (write_type == 4)
	{
		dest->val = source->val;
		dest->pmi = source->npmi + shift;
	}
	else
	{
		dest->val = source->pmi + shift;
		dest->pmi = source->npmi + shift;
	}
}

void copy_np_2(COLLOC_NP *source, COLLOC_NP *dest, double shift)
{
	dest->row = source->row;
	dest->col = source->col;
	dest->val = source->val;
	dest->pmi = source->pmi + shift;
	dest->npmi = source->npmi + shift;
}

int write_chunk(COLLOC_NP *buffer, int count, int write_type, double cutp, double cutn, double shift, FILE *fid)
{
	int i, idx = 0, cnt_write = 0;
	for (i = 0; i < count; i++)
		if (buffer[i].pmi > cutp && buffer[i].npmi > cutn)
			cnt_write++;

	if (write_type == 0 || write_type == 1 || write_type == 2)
	{
		COLLOC *buffer_w = (COLLOC*)malloc(sizeof(COLLOC) * cnt_write);
		for (i = 0; i < count; i++)
		{
			if (buffer[i].pmi > cutp && buffer[i].npmi > cutn)
			{
				copy_np_0(&buffer[i], &buffer_w[idx], shift, write_type);
				idx++;
			}
		}
		fwrite(buffer_w, sizeof(COLLOC), cnt_write, fid);
		fflush(fid);
		free(buffer_w); buffer_w = NULL;
	}
	else if (write_type == 6)
	{
		COLLOC_NP *buffer_w = (COLLOC_NP*)malloc(sizeof(COLLOC_NP) * cnt_write);
		for (i = 0; i < count; i++)
		{
			if (buffer[i].pmi > cutp && buffer[i].npmi > cutn)
			{
				copy_np_2(&buffer[i], &buffer_w[idx], shift);
				idx++;
			}
		}
		fwrite(buffer_w, sizeof(COLLOC_NP), cnt_write, fid);
		fflush(fid);
		free(buffer_w); buffer_w = NULL;
	}
	else
	{
		COLLOC_P *buffer_w = (COLLOC_P*)malloc(sizeof(COLLOC_P) * cnt_write);
		for (i = 0; i < count; i++)
		{
			if (buffer[i].pmi > cutp && buffer[i].npmi > cutn)
			{
				copy_np_1(&buffer[i], &buffer_w[idx], shift, write_type);
				idx++;
			}
		}
		fwrite(buffer_w, sizeof(COLLOC_P), cnt_write, fid);
		fflush(fid);
		free(buffer_w); buffer_w = NULL;
	}

	return cnt_write;
}

void pmi(char *input_file, char *output_file, double cutp, double cutn, double shift, double smooth, int smooth_type, double context_dist_smooth, int write_type)
{
	int i, n, nrow = 0, ncol = 0, nrec = 0, nrec_out = 0;
	scan_rowcol(input_file, &nrow, &ncol, &nrec);

	double sumsum = 0.0, minpmi = DBL_MAX, maxpmi = -DBL_MAX;
	double *sum_row = (double*)malloc(sizeof(double) * nrow);
	double *sum_col = (double*)malloc(sizeof(double) * ncol);
	if (sum_row == NULL || sum_col == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }
	calc_stats(input_file, sum_row, nrow, sum_col, ncol, &sumsum, smooth, smooth_type, context_dist_smooth);

	COLLOC *buffer_r = (COLLOC*)malloc(sizeof(COLLOC) * BUFF_SIZE);
	COLLOC_NP *buffer_w = (COLLOC_NP*)malloc(sizeof(COLLOC_NP) * BUFF_SIZE);
	if (buffer_r == NULL || buffer_w == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }

	FILE *fr = fopen(input_file, "rb");
	FILE *fw = fopen(output_file, "wb");
	if (fr == NULL || fw == NULL) { printf("Error: could not open input/output files.\r\n"); exit(EXIT_FAILURE); }

	do
	{
		n = fread(buffer_r, sizeof(COLLOC), BUFF_SIZE, fr);
		for (i = 0; i < n; i++)
		{
			copy_0_np(&buffer_r[i], &buffer_w[i]);

			buffer_w[i].pmi = log2((buffer_w[i].val + smooth) / (sumsum * sum_row[buffer_w[i].row] * sum_col[buffer_w[i].col]));
			buffer_w[i].npmi = -buffer_w[i].pmi / log2((buffer_w[i].val + smooth) / sumsum);

			if (buffer_w[i].pmi < minpmi)
				minpmi = buffer_w[i].pmi;
			if (buffer_w[i].pmi > maxpmi)
				maxpmi = buffer_w[i].pmi;
		}
		nrec_out += write_chunk(buffer_w, n, write_type, cutp, cutn, shift, fw);
	} while (n == BUFF_SIZE);
	fclose(fr);
	fclose(fw);

	printf("Minimum PMI: %5.2f\r\n", minpmi);
	printf("Maximum PMI: %5.2f\r\n", maxpmi);
	printf("%d records has been read from the input file.\r\n", nrec);
	printf("%d records has been written to the output file.\r\n\r\n", nrec_out);

	free(sum_row); sum_row = NULL;
	free(sum_col); sum_col = NULL;
	free(buffer_r); buffer_r = NULL;
	free(buffer_w); buffer_w = NULL;
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
	printf("\r\nTool to filter cooccurrences based on Pointwise Mutual Information (PMI) or Normalized PMI (NPMI).\r\n");
	printf("Author: Behrouz Haji Soleimani (behrouz.hajisoleimani@dal.ca)\r\n\r\n");
	printf("Usage: ./pmi -i <input_file> [-o <output_file>] [-cp <double>] [-cn <double>] [-w <int>] \r\n\r\n");
	printf("Example usage:\r\n");
	printf("./pmi -i cooccur.bin -o cooccur_pmi.bin -cp 0 -w 0\r\n\r\n");
	printf("Options:\r\n");
	printf("    -i <file>\r\n");
	printf("        Path to the input file containing the cooccurrences or any sparse matrix\r\n\r\n");
	printf("    -o <file>\r\n");
	printf("        Path to the output file (filtered records will be written here!)\r\n\r\n");
	printf("    -cp <double>\r\n");
	printf("        PMI cutoff threshold. Only records with PMI larger than <double> will be written to the output. Default: -1.79e+308 (don't filter!)\r\n\r\n");
	printf("    -cn <double>\r\n");
	printf("        NPMI cutoff threshold. Only records with NPMI larger than <double> will be written to the output. Default: -1.79e+308 (don't filter!)\r\n\r\n");
	printf("    -sh <double>\r\n");
	printf("        Add a constant to all PMI/NPMI values (shifted PMI). Default: 0.0\r\n\r\n");
	printf("    -sm <double>\r\n");
	printf("        Laplace smoothing for all elements of the matrix. Adds <double> to every element. Default: 0.0\r\n\r\n");
	printf("    -sm2 <double>\r\n");
	printf("        Laplace smoothing for non-zero elements of the matrix. Adds <double> to non-zero elements. sm2 suppresses sm. Default: 0.0\r\n\r\n");
	printf("    -w <int>\r\n");
	printf("        Can be 0, 1, 2, 3, 4, 5, or 6 specifying what information to be written to the output.\r\n");
	printf("        0 (default): same as input (row_idx, col_idx, value)\r\n");
	printf("        1: (row_idx, col_idx, PMI)\r\n");
	printf("        2: (row_idx, col_idx, NPMI)\r\n");
	printf("        3: (row_idx, col_idx, value, PMI)\r\n");
	printf("        4: (row_idx, col_idx, value, NPMI)\r\n");
	printf("        5: (row_idx, col_idx, PMI, NPMI)\r\n");
	printf("        6: (row_idx, col_idx, value, PMI, NPMI)\r\n\r\n");
	exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
	int i, write_type = DEFAULT_WRITE, smooth_type = DEFAULT_SMOOTH_TYPE;
	double cutp = DEFAULT_CUTP, cutn = DEFAULT_CUTN, shift = DEFAULT_SHIFT, smooth = DEFAULT_SMOOTH, context_dist_smooth = DEFAULT_CONTEXT_SMOOTH;
	char *input_file = malloc(sizeof(char) * MAX_FILE_NAME);
	char *output_file = malloc(sizeof(char) * MAX_FILE_NAME);
	if (input_file == NULL || output_file == NULL) { printf("Memory allocation failed.\r\n"); exit(EXIT_FAILURE); }

	if ((i = find_arg((char *)"-i", argc, argv)) > 0 || (i = find_arg((char *)"-input", argc, argv)) > 0)
		strcpy(input_file, argv[i + 1]);
	else
		usage();

	if ((i = find_arg((char *)"-o", argc, argv)) > 0 || (i = find_arg((char *)"-output", argc, argv)) > 0)
		strcpy(output_file, argv[i + 1]);
	else
		strcpy(output_file, (char*)"pmi_out.bin");

	if ((i = find_arg((char *)"-cp", argc, argv)) > 0 || (i = find_arg((char *)"-pmicutoff", argc, argv)) > 0)
		cutp = atof(argv[i + 1]);
	if ((i = find_arg((char *)"-cn", argc, argv)) > 0 || (i = find_arg((char *)"-npmicutoff", argc, argv)) > 0)
		cutn = atof(argv[i + 1]);
	if ((i = find_arg((char *)"-sh", argc, argv)) > 0 || (i = find_arg((char *)"-shift", argc, argv)) > 0)
		shift = atof(argv[i + 1]);
	if ((i = find_arg((char *)"-w", argc, argv)) > 0 || (i = find_arg((char *)"-write", argc, argv)) > 0)
		write_type = atoi(argv[i + 1]);
	if ((i = find_arg((char *)"-cs", argc, argv)) > 0 || (i = find_arg((char *)"-contextsmooth", argc, argv)) > 0)
		context_dist_smooth = atof(argv[i + 1]);
	if ((i = find_arg((char *)"-sm2", argc, argv)) > 0 || (i = find_arg((char *)"-smoothnnz", argc, argv)) > 0)
	{
		smooth = atof(argv[i + 1]);
		smooth_type = 2;
	}
	else if ((i = find_arg((char *)"-sm", argc, argv)) > 0 || (i = find_arg((char *)"-sm1", argc, argv)) > 0 || (i = find_arg((char *)"-smooth", argc, argv)) > 0 || (i = find_arg((char *)"-smoothall", argc, argv)) > 0)
	{
		smooth = atof(argv[i + 1]);
		smooth_type = 1;
	}

	printf("INITIALIZING ...\r\n");
	printf("PMI cutoff: %.5g\r\n", cutp);
	printf("NPMI cutoff: %.5g\r\n", cutn);
	printf("PMI/NPMI values will be shifted by: %lf\r\n", shift);
	if (context_dist_smooth > DBL_MIN) printf("Context distribution will be smoothed by alpha: %lf\r\n", context_dist_smooth);

	pmi(input_file, output_file, cutp, cutn, shift, smooth, smooth_type, context_dist_smooth, write_type);
	return EXIT_SUCCESS;
}
