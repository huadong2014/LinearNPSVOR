#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include "linear.h"
#include <time.h>//modification
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

void print_null(const char *s) {}

void exit_with_help()
{
	printf(
	"Usage: train [options] training_set_file [model_file]\n"
	"options:\n"
	"-s type : set type of solver (default 1)\n"
	"  for multi-class classification\n"
	"	 0 -- L2-regularized logistic regression (primal)\n"
	"	 1 -- L2-regularized L2-loss support vector classification (dual)\n"
	"	 2 -- L2-regularized L2-loss support vector classification (primal)\n"
	"	 3 -- L2-regularized L1-loss support vector classification (dual)\n"
	"	 4 -- support vector classification by Crammer and Singer\n"
	"	 5 -- L1-regularized L2-loss support vector classification\n"
	"	 6 -- L1-regularized logistic regression\n"
	"	 7 -- L2-regularized logistic regression (dual)\n"
	"	 8 -- L2-regularized SVOR\n"//my modification
	"    9 -- L2-regularized NPSVOR\n"
	"    10 -- L2-regularized SVMOP\n"
	"  for regression\n"
	"	11 -- L2-regularized L2-loss support vector regression (primal)\n"
	"	12 -- L2-regularized L2-loss support vector regression (dual)\n"
	"	13 -- L2-regularized L1-loss support vector regression (dual)\n"
	"  ordinal regression\n"	
	"	14 -- CSOR\n"
	"	15 -- SSVOR\n"
	"	16 -- SNPSVOR\n"	
	"-c cost : set the parameter C (default 1)\n"
	"-o cost one: set the parameter C1 for NPSVOR(defult = C)\n"
	"-t cost two: set the parameter C2 for NPSVOR(defult = C)\n"
	"-g grid seach C1 and C2: g=1 find C1=C2, g=2 find C1!=C2\n"	
	"-r rho: parameter of ADMM for SVOR\n"
	"-l weight loss for SVOR,|k-y|^w, w in {0,1,2}\n"
	"-m select the algorithm for svor\n"
	"-p epsilon : set the epsilon in loss function of SVR (default 0.1)\n"
	"-e epsilon : set tolerance of termination criterion\n"
	"	-s 0 and 2\n"
	"		|f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,\n"
	"		where f is the primal function and pos/neg are # of\n"
	"		positive/negative data (default 0.01)\n"
	"	-s 11\n"
	"		|f'(w)|_2 <= eps*|f'(w0)|_2 (default 0.001)\n"
	"	-s 1, 3, 4, and 7\n"
	"		Dual maximal violation <= eps; similar to libsvm (default 0.1)\n"
	"	-s 5 and 6\n"
	"		|f'(w)|_1 <= eps*min(pos,neg)/l*|f'(w0)|_1,\n"
	"		where f is the primal function (default 0.01)\n"
	"	-s 12 and 13\n"
	"		|f'(alpha)|_1 <= eps |f'(alpha0)|,\n"
	"		where f is the dual function (default 0.1)\n"
	"-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)\n"
	"-wi weight: weights adjust the parameter C of different classes (see README for details)\n"
	"-v n: n-fold cross validation mode\n"
	"-C : find parameter C (only for -s 0 and 2)\n"
	"-q : quiet mode (no outputs)\n"
	"-P :performance with time\n"
	);
	exit(1);
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);
void read_problem(const char *filename);
void do_cross_validation(const char *filename);
void do_find_parameter_C(const char *filename,const char *model_file_name);
void do_find_parameter_npsvor(const char *filename,const char *model_file_name);
void set_print_string_function(void (*print_func)(const char*));
void do_cross_validation_npsvor(const char *input_file_name);

struct feature_node *x_space;
struct parameter param;
struct problem prob;
struct model* model_;
int flag_cross_validation;
int flag_find_C;
int flag_C_specified;
int flag_solver_specified;
int nr_fold;
double bias;

int main(int argc, char **argv)
{
	char input_file_name[1024];
	char model_file_name[1024];
	const char *error_msg;
	parse_command_line(argc, argv, input_file_name, model_file_name);
	read_problem(input_file_name);
	
	error_msg = check_parameter(&prob,&param);

	if(error_msg)
	{
		fprintf(stderr,"ERROR: %s\n",error_msg);
		exit(1);
	}
    

	if (flag_find_C)
	{
		if(param.solver_type == L2R_NPSVOR && param.g ==2)
			do_find_parameter_npsvor(input_file_name,model_file_name);
		else
			do_find_parameter_C(input_file_name,model_file_name);	
	}
	else if(flag_cross_validation)
	{
		
		// do_cross_validation(input_file_name);
		do_cross_validation(input_file_name);
	}
	else
	{
		//fprintf(stderr,"Test:param.solver_type  %d\n",param.solver_type);
		model_=train(&prob, &param);
				// for(i=0;i<sub_prob.n;i++)
				// 	info("w%.6f\n",model_->w[i]);
				// for(i=0;i<nr_class-1;i++)
				// 	info("b %.6f\n",model_->b[i]);			
		if(save_model(model_file_name, model_))
		{
			fprintf(stderr,"can't save model to file %s\n",model_file_name);
			exit(1);
		}
		free_and_destroy_model(&model_);
	}
	destroy_param(&param);
	free(prob.y);
	free(prob.x);
	free(x_space);
	free(line);

	return 0;
}

static const char *solver_type_table[]=
{
	"L2R_LR", "L2R_L2LOSS_SVC_DUAL", "L2R_L2LOSS_SVC", "L2R_L1LOSS_SVC_DUAL", "MCSVM_CS",
	"L1R_L2LOSS_SVC", "L1R_LR", "L2R_LR_DUAL",
	"L2R_SVOR", "L2R_NPSVOR", "L2R_SVMOP",//My modification
	"L2R_L2LOSS_SVR", "L2R_L2LOSS_SVR_DUAL", "L2R_L1LOSS_SVR_DUAL", "SSVOR","NPSSVOR",NULL
};

void do_find_parameter_C(const char *input_file_name,const char *model_file_name)
{
	double start_C, best_C, best_acc_rate, best_mae_rate;
	// double max_C = param.C;
	double max_C = 8;	
	clock_t start, stop;
	double cvtime,traintime;
	if (flag_C_specified)
		start_C = param.C;
	else
		start_C = -1.0;
	// printf("Doing parameter search with %d-fold cross validation.\n", nr_fold);
	start=clock();
	find_parameter_C(&prob, &param, nr_fold, start_C, max_C, &best_C, &best_acc_rate,&best_mae_rate, input_file_name);
	stop=clock();
	cvtime = (double)(stop-start)/CLOCKS_PER_SEC;
	printf("Best C = %g  CV acc = %g CV mae = %g%%\n",best_C, best_acc_rate, best_mae_rate);
	param.C = best_C;
	if(param.solver_type == L2R_NPSVOR || param.solver_type == MCSVM_CS)
	{
		param.C1 = param.C;
		param.C2 = param.C;	
	}		

	// printf("%g\n",log(param.C)/log(2.0));
	do_cross_validation(input_file_name);
	start=clock();
 	model_=train(&prob, &param);
 	stop=clock();
	traintime = (double)(stop-start)/CLOCKS_PER_SEC;
	if(save_model(model_file_name, model_))
		{
			fprintf(stderr,"can't save model to file %s\n",model_file_name);
			exit(1);
		}
	free_and_destroy_model(&model_);

	char file_name[1024];
	if(param.solver_type == MCSVM_CS && param.npsvor == 7)
		sprintf(file_name,"%s_%s%g_out.log",solver_type_table[15],input_file_name,prob.bias);
	else if(param.solver_type == MCSVM_CS && param.npsvor == 3)
		sprintf(file_name,"%s_%s%g_out.log",solver_type_table[14],input_file_name,prob.bias);
	else
		sprintf(file_name,"%s_%s%g_out.log",solver_type_table[param.solver_type],input_file_name,prob.bias);

	// sprintf(file_name,"%s_%s%g_out.log",solver_type_table[param.solver_type],input_file_name,prob.bias);
	FILE * out = fopen(file_name,"a+t");
    fprintf(out, "%g %g\n",cvtime,traintime);	    
	if (NULL != out)
	   fclose(out) ;	// clear the old file.	 
}

void do_find_parameter_npsvor(const char *input_file_name,const char *model_file_name)
{
	double start_C, best_C1, best_C2, best_acc_rate, best_mae_rate;
	double max_C = 8;
	clock_t start, stop;
	double cvtime,traintime;
	if (flag_C_specified)
		start_C = param.C;
	else
		start_C = -1.0;
	printf("Doing parameter search with %d-fold cross validation.\n", nr_fold);
	start=clock();	
	find_parameter_npsvor(&prob, &param, nr_fold, start_C, max_C, &best_C1, &best_C2, &best_acc_rate,&best_mae_rate, input_file_name);
	stop=clock();
	cvtime = (double)(stop-start)/CLOCKS_PER_SEC;	
	if(param.solver_type == L2R_NPSVOR)
	{
		param.C1 = best_C1;
		param.C2 = best_C2;	
	}		

	// printf("%g\n",log(param.C)/log(2.0));
	do_cross_validation_npsvor(input_file_name);
	
	start=clock();
 	model_=train(&prob, &param);
 	stop=clock();
	traintime = (double)(stop-start)/CLOCKS_PER_SEC;
	if(save_model(model_file_name, model_))
		{
			fprintf(stderr,"can't save model to file %s\n",model_file_name);
			exit(1);
		}
	free_and_destroy_model(&model_);

	char file_name[1024];
	sprintf(file_name,"%s_%s%gc1c2_out.log",solver_type_table[param.solver_type],input_file_name,prob.bias);
	FILE * out = fopen(file_name,"a+t");
    fprintf(out, "%g %g\n",cvtime,traintime);	        
	if (NULL != out)
	   fclose(out) ;	// clear the old file.	
}

void do_cross_validation_npsvor(const char *input_file_name)
{
	int i;
	int total_correct = 0;
	double total_error = 0;
	double total_sqerror = 0;
	double *target = Malloc(double, prob.l);
	double cvtime;
	char file_name[1024];
	sprintf(file_name,"%s_%s%gc1c2_cv.log",solver_type_table[param.solver_type],input_file_name,prob.bias);
	FILE * log = fopen(file_name,"w");
	sprintf(file_name,"%s_%s%gc1c2_out.log",solver_type_table[param.solver_type],input_file_name,prob.bias);
	FILE * out = fopen(file_name,"at+");
	set_print_string_function(&print_null);
	clock_t start, stop;
	start=clock();
	cross_validation(&prob,&param,nr_fold,target);
	stop=clock();
	cvtime = (double)(stop-start)/CLOCKS_PER_SEC;	
	for(i=0;i<prob.l;i++)
		{if(target[i] == prob.y[i])
			++total_correct;
		total_error += fabs(target[i]-prob.y[i]);
		total_sqerror += (target[i]-prob.y[i])*(target[i]-prob.y[i]);
		}
		//printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
	// printf(" %g %g\n",1.0*total_correct/prob.l,total_error/prob.l);	

	int nr_class = 0;//number of classes
	int max_nr_class = 16;//max number of classes
	int *label = new int[max_nr_class];//category of labels
	int this_label;
	int j;
	for(i=0;i<prob.l;i++)
	{
		this_label = (int)prob.y[i];
		for(j=0;j<nr_class;j++)
		{
			if(this_label == label[j])
				break;
		}
		if(j == nr_class)
		{
			label[nr_class] = this_label;
			++nr_class;
		}
	}
	int *ConfusionMatrix = new int[nr_class*nr_class];
	memset(ConfusionMatrix,0,sizeof(int)*(nr_class*nr_class));
	for(i=0;i<prob.l;i++)
		ConfusionMatrix[(int)(prob.y[i]-1)*nr_class+ (int)target[i]-1] += 1; 

	fprintf(out, "%g %g %g %g %g %g %g \n", param.p,
		param.C1, param.C2, 1.0*total_correct/prob.l,total_error/prob.l,total_sqerror/prob.l, cvtime);
	printf("%g %g %g %g %g %g %g \n", param.p,
		param.C1, param.C2, 1.0*total_correct/prob.l,total_error/prob.l,total_sqerror/prob.l, cvtime);	
	fprintf(out, "ConfusionMatrix:\n");
	for(i=0;i<nr_class*nr_class;i++)
		{
			fprintf(out, "%d\t", ConfusionMatrix[i]);
			if((i+1)%nr_class==0)
				fprintf(out, "\n");
		}

	for(i=0;i<prob.l;i++)
	   fprintf(log,"%g %g\n", prob.y[i], target[i]); 
	if (NULL != log)
	   fclose(log) ;	// clear the old file.	                     
	free(target);
	if (NULL != out)
	   fclose(out) ;	// clear the old file.
	free(ConfusionMatrix);
	free(label);		
}


void do_cross_validation(const char *input_file_name)
{
	int i;
	int total_correct = 0;
	double total_error = 0;
	double total_sqerror = 0;
	double *target = Malloc(double, prob.l);
	double cvtime;
	char file_name[1024];


	FILE * log = fopen(file_name,"w");
	if(param.solver_type == MCSVM_CS && param.npsvor == 7)
	{
		sprintf(file_name,"%s_%s%g_cv.log",solver_type_table[15],input_file_name,prob.bias);	
		sprintf(file_name,"%s_%s%g_out.log",solver_type_table[15],input_file_name,prob.bias);		
	}
	else if(param.solver_type == MCSVM_CS && param.npsvor == 3)
	{
		sprintf(file_name,"%s_%s%g_cv.log",solver_type_table[15],input_file_name,prob.bias);	
		sprintf(file_name,"%s_%s%g_out.log",solver_type_table[15],input_file_name,prob.bias);		
	}
	else
	{
		sprintf(file_name,"%s_%s%g_cv.log",solver_type_table[param.solver_type],input_file_name,prob.bias);	
		sprintf(file_name,"%s_%s%g_out.log",solver_type_table[param.solver_type],input_file_name,prob.bias);			
	}


	FILE * out = fopen(file_name,"wt+");
	set_print_string_function(&print_null);
	clock_t start, stop;
	start=clock();
	cross_validation(&prob,&param,nr_fold,target);
	stop=clock();
	cvtime = (double)(stop-start)/CLOCKS_PER_SEC;	
	for(i=0;i<prob.l;i++)
		{if(target[i] == prob.y[i])
			++total_correct;
		total_error += fabs(target[i]-prob.y[i]);
		total_sqerror += (target[i]-prob.y[i])*(target[i]-prob.y[i]);
		}
		//printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
	// printf(" %g %g\n",1.0*total_correct/prob.l,total_error/prob.l);	

	int nr_class = 0;//number of classes
	int max_nr_class = 16;//max number of classes
	int *label = new int[max_nr_class];//category of labels
	int this_label;
	int j;
	for(i=0;i<prob.l;i++)
	{
		this_label = (int)prob.y[i];
		for(j=0;j<nr_class;j++)
		{
			if(this_label == label[j])
				break;
		}
		if(j == nr_class)
		{
			label[nr_class] = this_label;
			++nr_class;
		}
	}
	int *ConfusionMatrix = new int[nr_class*nr_class];
	memset(ConfusionMatrix,0,sizeof(int)*(nr_class*nr_class));
	for(i=0;i<prob.l;i++)
		ConfusionMatrix[(int)(prob.y[i]-1)*nr_class+ (int)target[i]-1] += 1; 

	fprintf(out, "%g %g %g %g %g %g ", param.p,
		param.C, 1.0*total_correct/prob.l,total_error/prob.l,total_sqerror/prob.l, cvtime);
	printf("%g, %g, %g %g %g \n", param.p,
		param.C, 1.0*total_correct/prob.l,total_error/prob.l, cvtime);	
	// fprintf(out, "ConfusionMatrix:\n");
	// for(i=0;i<nr_class*nr_class;i++)
	// 	{
	// 		fprintf(out, "%d\t", ConfusionMatrix[i]);
	// 		if((i+1)%nr_class==0)
	// 			fprintf(out, "\n");
	// 	}

	for(i=0;i<prob.l;i++)
	   fprintf(log,"%g %g\n", prob.y[i], target[i]); 
	if (NULL != log)
	   fclose(log) ;	// clear the old file.	                     
	free(target);
	if (NULL != out)
	   fclose(out) ;	// clear the old file.
	free(ConfusionMatrix);
	free(label);		
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{
	int i;
	void (*print_func)(const char*) = NULL;	// default printing to stdout

	// default values
	param.solver_type = L2R_L2LOSS_SVC_DUAL;
	param.C = 1;
	param.eps = INF; // see setting below
	param.p = 0.1;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	param.init_sol = NULL;
	param.WithTime = 0;
/*
* -------------------my modification---------------------------
*/
	param.rho = 1;
	param.wl = 0;
	param.svor = 1;
	param.npsvor = 1;
	param.C1 = 1;
	param.C2 = 1;
	param.g = 1;
/*
* -------------------my modification---------------------------
*/
	flag_cross_validation = 0;
	flag_C_specified = 0;
	flag_solver_specified = 0;
	flag_find_C = 0;
	bias = -1;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
/*
* -------------------my modification---------------------------
*/
			case 'r'://my parameter s;
				param.rho = atof(argv[i]);
				break;

			case 'l'://my parameter t;
				param.wl = atof(argv[i]);
				break;
			case 'm'://my parameter t;
				param.svor = atof(argv[i]);
				param.npsvor = atof(argv[i]);
				break;
			case 'o': // one
				param.C1 = atof(argv[i]);
				break;
			case 't': // two
				param.C2 = atof(argv[i]);
				break;
			case 'g': // two
				param.g = atof(argv[i]);
				break;				
/*
* -------------------my modification---------------------------
*/
			case 's':
				param.solver_type = atoi(argv[i]);
				flag_solver_specified = 1;
				break;

			case 'c':
				param.C = atof(argv[i]);
				param.C1 = param.C; //
				param.C2 = param.C; //
				flag_C_specified = 1;
				break;

			case 'p':
				param.p = atof(argv[i]);
				break;

			case 'e':
				param.eps = atof(argv[i]);
				break;

			case 'B':
				bias = atof(argv[i]);
				break;

			case 'w':
				++param.nr_weight;
				param.weight_label = (int *) realloc(param.weight_label,sizeof(int)*param.nr_weight);
				param.weight = (double *) realloc(param.weight,sizeof(double)*param.nr_weight);
				param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
				param.weight[param.nr_weight-1] = atof(argv[i]);
				break;

			case 'v':
				flag_cross_validation = 1;
				nr_fold = atoi(argv[i]);
				if(nr_fold < 2)
				{
					fprintf(stderr,"n-fold cross validation: n must >= 2\n");
					exit_with_help();
				}
				break;

			case 'q':
				print_func = &print_null;
				i--;
				break;

			case 'C':
				flag_find_C = 1;
				i--;
				break;
			case 'P':
				param.WithTime = 1;
				i--;
				break;
			default:
				fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}

	set_print_string_function(print_func);

	// determine filenames
	if(i>=argc)
		exit_with_help();

	strcpy(input_file_name, argv[i]);

	if(i<argc-1)
		strcpy(model_file_name,argv[i+1]);
	else
	{
		char *p = strrchr(argv[i],'/');
		if(p==NULL)
			p = argv[i];
		else
			++p;
		sprintf(model_file_name,"%s.model",p);
	}

	// default solver for parameter selection is L2R_L2LOSS_SVC
	if(flag_find_C)
	{
		if(!flag_cross_validation)
			nr_fold = 5;
		if(!flag_solver_specified)
		{
			fprintf(stderr, "Solver not specified. Using -s 2\n");
			param.solver_type = L2R_L2LOSS_SVC;
		}
		// else if(param.solver_type != L2R_LR && param.solver_type != L2R_L2LOSS_SVC)
		// {
		// 	fprintf(stderr, "Warm-start parameter search only available for -s 0 and -s 2\n");
		// 	exit_with_help();
		// }
	}
	if(param.eps == INF)
	{
		switch(param.solver_type)
		{
/*
* -------------------my modification---------------------------
*/
			case L2R_SVOR:
				param.eps = 0.1;
				param.rho = 1;
				break;
			case L2R_NPSVOR:
				param.eps = 0.1;
				break;
/*
* -------------------my modification---------------------------
*/
			case L2R_LR:
			case L2R_L2LOSS_SVC:
				param.eps = 0.01;
				break;
			case L2R_L2LOSS_SVR:
				param.eps = 0.001;
				break;
			case L2R_L2LOSS_SVC_DUAL:
			case L2R_L1LOSS_SVC_DUAL://in this case, eps is set to 0.1
			case L2R_SVMOP://in this case, eps is set to 0.1			
				//param.eps = 0.1;
				//break;
			case MCSVM_CS:
			case L2R_LR_DUAL:
				param.eps = 0.1;
				break;
			case L1R_L2LOSS_SVC:
			case L1R_LR:
				param.eps = 0.01;
				break;
			case L2R_L1LOSS_SVR_DUAL:
			case L2R_L2LOSS_SVR_DUAL:
				param.eps = 0.1;
				break;
		}
	}
}

// read in a problem (in libsvm format)
void read_problem(const char *filename)
{
	int max_index, inst_max_index, i;
	size_t elements, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	prob.l = 0;
	elements = 0;
	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			elements++;
		}
		elements++; // for bias term
		prob.l++;
	}
	rewind(fp);

	prob.bias=bias;

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct feature_node *,prob.l);
	x_space = Malloc(struct feature_node,elements+prob.l);

	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)
	{
		inst_max_index = 0; // strtol gives 0 if wrong format
		readline(fp);
		prob.x[i] = &x_space[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		prob.y[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				exit_input_error(i+1);
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);

			++j;
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;

		if(prob.bias >= 0)
			x_space[j++].value = prob.bias;

		x_space[j++].index = -1;
	}

	if(prob.bias >= 0)
	{
		prob.n=max_index+1;
		for(i=1;i<prob.l;i++)
			(prob.x[i]-2)->index = prob.n;
		x_space[j-2].index = prob.n;
	}
	else
		prob.n=max_index;

	fclose(fp);
}
