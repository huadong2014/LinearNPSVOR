
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <locale.h>
#include <time.h>//modification
#include "linear.h"
#include "tron.h"
typedef signed char schar;
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

#ifdef __cplusplus
extern "C" {
#endif

extern double dnrm2_(int *, double *, int *);
extern double ddot_(int *, double *, int *, double *, int *);
extern int daxpy_(int *, double *, double *, int *, double *, int *);
extern int dscal_(int *, double *, double *, int *);

#ifdef __cplusplus
}
#endif


static void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}
static void print_null(const char *s) {}

static void (*liblinear_print_string) (const char *) = &print_string_stdout;

#if 1
static void info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*liblinear_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif
class sparse_operator
{
public:
	static double nrm2_sq(const feature_node *x)//norm2-square
	{
		double ret = 0;
		while(x->index != -1)
		{
			ret += x->value*x->value;
			x++;
		}
		return (ret);
	}

	static double dot(const double *s, const feature_node *x)
	{
		double ret = 0;
		while(x->index != -1)
		{
			ret += s[x->index-1]*x->value;
			x++;
		}
		return (ret);
	}

	static void axpy(const double a, const feature_node *x, double *y)//a*x+y
	{
		while(x->index != -1)
		{
			y[x->index-1] += a*x->value;
			x++;
		}
	}

	static void xxy(const feature_node *x, double *y)//xx+y
	{
		while(x->index != -1)
		{
			y[x->index-1] += x->value*x->value;
			x++;
		}
	}

	static void xxyc(const feature_node *x, double C, double *y)//xx+y
	{
		while(x->index != -1)
		{
			y[x->index-1] += C*x->value*x->value;
			x++;
		}
	}		
};

class l2r_lr_fun: public function
{
public:
	l2r_lr_fun(const problem *prob, double *C);
	~l2r_lr_fun();

	double fun(double *w);
	void grad(double *w, double *g);
	void Hv(double *s, double *Hs);
	void PI(double *PI);
	int get_nr_variable(void);

private:
	void Xv(double *v, double *Xv);
	void XTv(double *v, double *XTv);

	double *C;
	double *z;
	double *D;
	const problem *prob;
};

l2r_lr_fun::l2r_lr_fun(const problem *prob, double *C)
{
	int l=prob->l;

	this->prob = prob;

	z = new double[l];
	D = new double[l];
	this->C = C;
}

l2r_lr_fun::~l2r_lr_fun()
{
	delete[] z;
	delete[] D;
}


double l2r_lr_fun::fun(double *w)
{
	int i;
	double f=0;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	Xv(w, z);

	for(i=0;i<w_size;i++)
		f += w[i]*w[i];
	f /= 2.0;
	for(i=0;i<l;i++)
	{
		double yz = y[i]*z[i];
		if (yz >= 0)
			f += C[i]*log(1 + exp(-yz));
		else
			f += C[i]*(-yz+log(1 + exp(yz)));
	}

	return(f);
}

void l2r_lr_fun::grad(double *w, double *g)
{
	int i;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	for(i=0;i<l;i++)
	{
		z[i] = 1/(1 + exp(-y[i]*z[i]));
		D[i] = z[i]*(1-z[i]);
		z[i] = C[i]*(z[i]-1)*y[i];
	}
	XTv(z, g);

	for(i=0;i<w_size;i++)
		g[i] = w[i] + g[i];
}

int l2r_lr_fun::get_nr_variable(void)
{
	return prob->n;
}

void l2r_lr_fun::Hv(double *s, double *Hs)
{
	int i;
	int l=prob->l;
	int w_size=get_nr_variable();
	double *wa = new double[l];
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		Hs[i] = 0;
	for(i=0;i<l;i++)
	{
		feature_node * const xi=x[i];
		wa[i] = sparse_operator::dot(s, xi);
		
		wa[i] = C[i]*D[i]*wa[i];

		sparse_operator::axpy(wa[i], xi, Hs);
	}
	for(i=0;i<w_size;i++)
		Hs[i] = s[i] + Hs[i];
	delete[] wa;
}

void l2r_lr_fun::Xv(double *v, double *Xv)
{
	int i;
	int l=prob->l;
	feature_node **x=prob->x;

	for(i=0;i<l;i++)
		Xv[i]=sparse_operator::dot(v, x[i]);
}

void l2r_lr_fun::XTv(double *v, double *XTv)
{
	int i;
	int l=prob->l;
	int w_size=get_nr_variable();
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		XTv[i]=0;
	for(i=0;i<l;i++)
		sparse_operator::axpy(v[i], x[i], XTv);
}

void l2r_lr_fun::PI(double *PI)
{
	int i;
	int w_size=get_nr_variable();
	for(i=0;i<w_size;i++)
	{		
		PI[i] = 1.0;
	}
}

class l2r_l2_svc_fun: public function
{
public:
	l2r_l2_svc_fun(const problem *prob, double *C);
	~l2r_l2_svc_fun();

	double fun(double *w);
	void grad(double *w, double *g);
	void Hv(double *s, double *Hs);
	void PI(double *PI);
	int get_nr_variable(void);

protected:
	void Xv(double *v, double *Xv);
	void subXTv(double *v, double *XTv);

	double *C;
	double *z;
	double *D;
	int *I;
	int sizeI;
	const problem *prob;
};

l2r_l2_svc_fun::l2r_l2_svc_fun(const problem *prob, double *C)
{
	int l=prob->l;

	this->prob = prob;

	z = new double[l];
	D = new double[l];
	I = new int[l];
	this->C = C;
}

l2r_l2_svc_fun::~l2r_l2_svc_fun()
{
	delete[] z;
	delete[] D;
	delete[] I;
}

double l2r_l2_svc_fun::fun(double *w)
{
	int i;
	double f=0;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	Xv(w, z);

	for(i=0;i<w_size;i++)
		f += w[i]*w[i];
	f /= 2.0;
	for(i=0;i<l;i++)
	{
		z[i] = y[i]*z[i];
		double d = 1-z[i];
		if (d > 0)
			f += C[i]*d*d;
	}

	return(f);
}

void l2r_l2_svc_fun::grad(double *w, double *g)
{
	int i;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	sizeI = 0;
	for (i=0;i<l;i++)
		if (z[i] < 1)
		{
			z[sizeI] = C[i]*y[i]*(z[i]-1);
			I[sizeI] = i;
			sizeI++;
		}
	subXTv(z, g);

	for(i=0;i<w_size;i++)
		g[i] = w[i] + 2*g[i];
}

int l2r_l2_svc_fun::get_nr_variable(void)
{
	return prob->n;
}

void l2r_l2_svc_fun::Hv(double *s, double *Hs)
{
	int i;
	int w_size=get_nr_variable();
	double *wa = new double[sizeI];
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		Hs[i]=0;
	for(i=0;i<sizeI;i++)
	{
		feature_node * const xi=x[I[i]];
		wa[i] = sparse_operator::dot(s, xi);
		
		wa[i] = C[I[i]]*wa[i];

		sparse_operator::axpy(wa[i], xi, Hs);
	}
	for(i=0;i<w_size;i++)
		Hs[i] = s[i] + 2*Hs[i];
	delete[] wa;
}

void l2r_l2_svc_fun::Xv(double *v, double *Xv)
{
	int i;
	int l=prob->l;
	feature_node **x=prob->x;

	for(i=0;i<l;i++)
		Xv[i]=sparse_operator::dot(v, x[i]);
}

void l2r_l2_svc_fun::subXTv(double *v, double *XTv)
{
	int i;
	int w_size=get_nr_variable();
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		XTv[i]=0;
	for(i=0;i<sizeI;i++)
		sparse_operator::axpy(v[i], x[I[i]], XTv);
}

void l2r_l2_svc_fun::PI(double *PI)
{
	int i;
	int w_size=get_nr_variable();
	for(i=0;i<w_size;i++)
	{		
		PI[i] = 1.0;
	}
}




class l2r_l2_svr_fun: public l2r_l2_svc_fun
{
public:
	l2r_l2_svr_fun(const problem *prob, double *C, double p);

	double fun(double *w);
	void grad(double *w, double *g);

private:
	double p;
};

l2r_l2_svr_fun::l2r_l2_svr_fun(const problem *prob, double *C, double p):
	l2r_l2_svc_fun(prob, C)
{
	this->p = p;
}

double l2r_l2_svr_fun::fun(double *w)
{
	int i;
	double f=0;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();
	double d;

	Xv(w, z);

	for(i=0;i<w_size;i++)
		f += w[i]*w[i];
	f /= 2;
	for(i=0;i<l;i++)
	{
		d = z[i] - y[i];
		if(d < -p)
			f += C[i]*(d+p)*(d+p);
		else if(d > p)
			f += C[i]*(d-p)*(d-p);
	}

	return(f);
}

void l2r_l2_svr_fun::grad(double *w, double *g)
{
	int i;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();
	double d;

	sizeI = 0;
	for(i=0;i<l;i++)
	{
		d = z[i] - y[i];

		// generate index set I
		if(d < -p)
		{
			z[sizeI] = C[i]*(d+p);
			I[sizeI] = i;
			sizeI++;
		}
		else if(d > p)
		{
			z[sizeI] = C[i]*(d-p);
			I[sizeI] = i;
			sizeI++;
		}

	}
	subXTv(z, g);

	for(i=0;i<w_size;i++)
		g[i] = w[i] + 2*g[i];
}


class l2r_l2_npsvor_fun: public function
{
public:
	l2r_l2_npsvor_fun(const problem *prob, const parameter *param, int k);
	~l2r_l2_npsvor_fun();

	double fun(double *w);
	void grad(double *w, double *g);
	void Hv(double *s, double *Hs);
	void PI(double *PI);
	int get_nr_variable(void);

private:
	void Xv(double *v, double *Xv);
	void subXTv(double *v, double *XTv);
	double C1;
	double C2;
	double *z;
	double *D;
	double p;
	double k;
	int *I;
	int sizeI;			
	const problem *prob;
	const parameter *param;	
};

l2r_l2_npsvor_fun::l2r_l2_npsvor_fun(const problem *prob, const parameter *param, int k)
{
	int l=prob->l;

	this->prob = prob;
	this->param = param;
	z = new double[l];
	D = new double[l];
	I = new int[l];	
	this->C1 = param->C1;
	this->C2 = param->C2;
	this->p = param->p;	
	this->k = k;		
}

l2r_l2_npsvor_fun::~l2r_l2_npsvor_fun()
{
	delete[] z;
	delete[] D;
	delete[] I;	
}


// double l2r_l2_npsvor_fun::fun(double *w)
// {
// 	int i;
// 	double f=0;
// 	double *y=prob->y;
// 	int l=prob->l;
// 	int w_size=get_nr_variable();
// 	double d;
// 	double yki;
// 	Xv(w, z);

// 	for(i=0;i<w_size;i++)
// 		f += w[i]*w[i];
// 	f /= 2.0;
// 	for(i=0;i<l;i++)
// 	{

// 		if(y[i]==k)
// 		{
// 			d = fabs(z[i])-p;
// 			if(d > 0)
// 				f += C1*d*d;			
// 		}
// 		else 
// 		{
// 			yki = (y[i]>k?1:-1);			
// 			d = 1 - yki*z[i];
// 			if(d>0)
// 				f += C2*d*d;					
// 		}
// 	}

// 	return(f);
// }


double l2r_l2_npsvor_fun::fun(double *w)
{
	int i;
	double f=0;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();
	double d;
	double yki;
	Xv(w, z);

	for(i=0;i<w_size;i++)
		f += w[i]*w[i];
	f /= 2.0;
	for(i=0;i<l;i++)
	{

		if(y[i]==k)
		{
			d = fabs(z[i])-p;
			if(d > 0)
				f += C1*d;			
		}
		else 
		{
			yki = (y[i]>k?1:-1);			
			d = 1 - yki*z[i];
			if(d>0)
				f += C2*d;					
		}
	}

	return(f);
}


void l2r_l2_npsvor_fun::grad(double *w, double *g)
{
	int i;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();
	double d;
	double yki;

	sizeI = 0;
	for (i=0;i<l;i++)
	{
		d = z[i];
		if(y[i]==k)
		{
			if(d < -p)
			{
				z[sizeI] = C1*(d+p);
				I[sizeI] = i;
				sizeI++;
			}
			else if(d > p)
			{
				z[sizeI] = C1*(d-p);
				I[sizeI] = i;
				sizeI++;
			}			
		}
		else
		{
			yki = (y[i]>k?1:-1);
			if (yki*z[i] < 1)
			{
				z[sizeI] = C2*(z[i]-yki);
				I[sizeI] = i;
				sizeI++;
			}		
		}
	}


	subXTv(z, g);

	for(i=0;i<w_size;i++)
		g[i] = w[i] + 2*g[i];
}

int l2r_l2_npsvor_fun::get_nr_variable(void)
{
	return prob->n;
}

void l2r_l2_npsvor_fun::Hv(double *s, double *Hs)
{
	int i;
	int w_size=get_nr_variable();
	double *wa = new double[sizeI];
	double *y=prob->y;	
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		Hs[i]=0;
	for(i=0;i<sizeI;i++)
	{
		feature_node * const xi=x[I[i]];
		wa[i] = sparse_operator::dot(s, xi);
		if(y[I[i]] == k)
			wa[i] = C1*wa[i];
		else
			wa[i] = C2*wa[i];
	
		sparse_operator::axpy(wa[i], xi, Hs);
	}
	for(i=0;i<w_size;i++)
		Hs[i] = s[i] + 2*Hs[i];
	delete[] wa;
}

void l2r_l2_npsvor_fun::Xv(double *v, double *Xv)
{
	int i;
	int l=prob->l;
	feature_node **x=prob->x;

	for(i=0;i<l;i++)
		Xv[i]=sparse_operator::dot(v, x[i]);
}

void l2r_l2_npsvor_fun::subXTv(double *v, double *XTv)
{
	int i;
	int w_size=get_nr_variable();
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		XTv[i]=0;
	for(i=0;i<sizeI;i++)
		sparse_operator::axpy(v[i], x[I[i]], XTv);
}


void l2r_l2_npsvor_fun::PI(double *PI)
{
	int i;
	feature_node **x=prob->x;
	int w_size=get_nr_variable();
	double *y=prob->y;		
	double *xx = new double[w_size];
	double P;
	memset(xx,0,sizeof(double)*w_size);	
	for(i=0;i<prob->l;i++)
	{
		if(y[i] == k)
			sparse_operator::xxyc(x[i], C1, xx);
		else
		 	sparse_operator::xxyc(x[i], C2, xx);			
	}
	for(i=0;i<w_size;i++)
	{		
		P = sqrt(1+xx[i]);
		PI[i] = 1.0/P;
	}
}



// A coordinate descent algorithm for 
// multi-class support vector machines by Crammer and Singer
//
//  min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i
//    s.t.     \alpha^m_i <= C^m_i \forall m,i , \sum_m \alpha^m_i=0 \forall i
// 
//  where e^m_i = 0 if y_i  = m,
//        e^m_i = 1 if y_i != m,
//  C^m_i = C if m  = y_i, 
//  C^m_i = 0 if m != y_i, 
//  and w_m(\alpha) = \sum_i \alpha^m_i x_i 
//
// Given: 
// x, y, C
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Appendix of LIBLINEAR paper, Fan et al. (2008)

#define GETI(i) ((int) prob->y[i])
// To support weights for instances, use GETI(i) (i)

class Solver_MCSVM_CS
{
	public:
		Solver_MCSVM_CS(const problem *prob, int nr_class, double *C, double eps=0.1, int max_iter=100000);
		~Solver_MCSVM_CS();
		void Solve(double *w);
	private:
		void solve_sub_problem(double A_i, int yi, double C_yi, int active_i, double *alpha_new);
		bool be_shrunk(int i, int m, int yi, double alpha_i, double minG);
		double *B, *C, *G;
		int w_size, l;
		int nr_class;
		int max_iter;
		double eps;
		const problem *prob;
};

Solver_MCSVM_CS::Solver_MCSVM_CS(const problem *prob, int nr_class, double *weighted_C, double eps, int max_iter)
{
	this->w_size = prob->n;
	this->l = prob->l;
	this->nr_class = nr_class;
	this->eps = eps;
	this->max_iter = max_iter;
	this->prob = prob;
	this->B = new double[nr_class];
	this->G = new double[nr_class];
	this->C = weighted_C;
}

Solver_MCSVM_CS::~Solver_MCSVM_CS()
{
	delete[] B;
	delete[] G;
}

int compare_double(const void *a, const void *b)
{
	if(*(double *)a > *(double *)b)
		return -1;
	if(*(double *)a < *(double *)b)
		return 1;
	return 0;
}

void Solver_MCSVM_CS::solve_sub_problem(double A_i, int yi, double C_yi, int active_i, double *alpha_new)
{
	int r;
	double *D;

	clone(D, B, active_i);
	if(yi < active_i)
		D[yi] += A_i*C_yi;
	qsort(D, active_i, sizeof(double), compare_double);

	double beta = D[0] - A_i*C_yi;
	for(r=1;r<active_i && beta<r*D[r];r++)
		beta += D[r];
	beta /= r;

	for(r=0;r<active_i;r++)
	{
		if(r == yi)
			alpha_new[r] = min(C_yi, (beta-B[r])/A_i);
		else
			alpha_new[r] = min((double)0, (beta - B[r])/A_i);
	}
	delete[] D;
}

bool Solver_MCSVM_CS::be_shrunk(int i, int m, int yi, double alpha_i, double minG)
{
	double bound = 0;
	if(m == yi)
		bound = C[GETI(i)];
	if(alpha_i == bound && G[m] < minG)
		return true;
	return false;
}

void Solver_MCSVM_CS::Solve(double *w)
{
	int tl = 0;
	int active_size = l -tl;
	clock_t start, stop;
	start=clock();


	if(prob->WithTime)
	{
		tl = (int)prob->l*3/10;
		active_size = l -tl;
					
	}	

	int i, m, s;
	int iter = 0;
	double *alpha =  new double[l*nr_class];
	double *alpha_new = new double[nr_class];
	int *index = new int[l];
	double *QD = new double[l];
	int *d_ind = new int[nr_class];
	double *d_val = new double[nr_class];
	int *alpha_index = new int[nr_class*l];
	int *y_index = new int[l];
	int *active_size_i = new int[l];
	double eps_shrink = max(10.0*eps, 1.0); // stopping tolerance for shrinking
	bool start_from_all = true;

	// Initial alpha can be set here. Note that 
	// sum_m alpha[i*nr_class+m] = 0, for all i=1,...,l-1
	// alpha[i*nr_class+m] <= C[GETI(i)] if prob->y[i] == m
	// alpha[i*nr_class+m] <= 0 if prob->y[i] != m
	// If initial alpha isn't zero, uncomment the for loop below to initialize w
	for(i=0;i<l*nr_class;i++)
		alpha[i] = 0;

	for(i=0;i<w_size*nr_class;i++)
		w[i] = 0;
	for(i=0;i<l;i++)
	{
		for(m=0;m<nr_class;m++)
			alpha_index[i*nr_class+m] = m;
		feature_node *xi = prob->x[i];
		QD[i] = 0;
		while(xi->index != -1)
		{
			double val = xi->value;
			QD[i] += val*val;

			// Uncomment the for loop if initial alpha isn't zero
			// for(m=0; m<nr_class; m++)
			//	w[(xi->index-1)*nr_class+m] += alpha[i*nr_class+m]*val;
			xi++;
		}
		active_size_i[i] = nr_class;
		y_index[i] = (int)prob->y[i];
		index[i] = i;
	}

	while(iter < max_iter)
	{
		double stopping = -INF;
		for(i=0;i<active_size;i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}
		for(s=0;s<active_size;s++)
		{

			i = index[s];
			double Ai = QD[i];
			double *alpha_i = &alpha[i*nr_class];
			int *alpha_index_i = &alpha_index[i*nr_class];

			if(Ai > 0)
			{
				for(m=0;m<active_size_i[i];m++)
					G[m] = 1;
				if(y_index[i] < active_size_i[i])
					G[y_index[i]] = 0;

				feature_node *xi = prob->x[i];
				while(xi->index!= -1)
				{
					double *w_i = &w[(xi->index-1)*nr_class];
					for(m=0;m<active_size_i[i];m++)
						G[m] += w_i[alpha_index_i[m]]*(xi->value);
					xi++;
				}

				double minG = INF;
				double maxG = -INF;
				for(m=0;m<active_size_i[i];m++)
				{
					if(alpha_i[alpha_index_i[m]] < 0 && G[m] < minG)
						minG = G[m];
					if(G[m] > maxG)
						maxG = G[m];
				}
				if(y_index[i] < active_size_i[i])
					if(alpha_i[(int) prob->y[i]] < C[GETI(i)] && G[y_index[i]] < minG)
						minG = G[y_index[i]];

				for(m=0;m<active_size_i[i];m++)
				{
					if(be_shrunk(i, m, y_index[i], alpha_i[alpha_index_i[m]], minG))
					{
						active_size_i[i]--;
						while(active_size_i[i]>m)
						{
							if(!be_shrunk(i, active_size_i[i], y_index[i],
											alpha_i[alpha_index_i[active_size_i[i]]], minG))
							{
								swap(alpha_index_i[m], alpha_index_i[active_size_i[i]]);
								swap(G[m], G[active_size_i[i]]);
								if(y_index[i] == active_size_i[i])
									y_index[i] = m;
								else if(y_index[i] == m)
									y_index[i] = active_size_i[i];
								break;
							}
							active_size_i[i]--;
						}
					}
				}

				if(active_size_i[i] <= 1)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}

				if(maxG-minG <= 1e-12)
					continue;
				else
					stopping = max(maxG - minG, stopping);

				for(m=0;m<active_size_i[i];m++)
					B[m] = G[m] - Ai*alpha_i[alpha_index_i[m]] ;

				solve_sub_problem(Ai, y_index[i], C[GETI(i)], active_size_i[i], alpha_new);
				int nz_d = 0;
				for(m=0;m<active_size_i[i];m++)
				{
					double d = alpha_new[m] - alpha_i[alpha_index_i[m]];
					alpha_i[alpha_index_i[m]] = alpha_new[m];
					if(fabs(d) >= 1e-12)
					{
						d_ind[nz_d] = alpha_index_i[m];
						d_val[nz_d] = d;
						nz_d++;
					}
				}

				xi = prob->x[i];
				while(xi->index != -1)
				{
					double *w_i = &w[(xi->index-1)*nr_class];
					for(m=0;m<nz_d;m++)
						w_i[d_ind[m]] += d_val[m]*xi->value;
					xi++;
				}
			}



			// performance with time

			if(prob->WithTime)
			{			
				if(s%(active_size/5)==0)
				///measures vs time
				{

					// calculate objective value
					double v = 0;
					int nSV = 0;
					for(i=0;i<w_size*nr_class;i++)
						v += w[i]*w[i];
					v = 0.5*v;
					for(i=0;i<l*nr_class;i++)
					{
						v += alpha[i];
						if(fabs(alpha[i]) > 0)
							nSV++;
					}
					for(i=0;i<l;i++)
						v -= alpha[i*nr_class+(int)prob->y[i]];

					int nr_w = nr_class;
					// int t;
					double mze=0;
					double mae = 0.0, mse = 0.0, T, diff,target_label,predict_y;	

					for(int t=l-tl;t<prob->l;t++)		
						{
							target_label = (double)prob->y[t]+1;
							predict_y = (double)predict_label_CrammerSinger(prob->x[t], w, prob->label, nr_w, w_size, nr_class);
						    //info("target_label=%.3f predict_y=%.3f\n",target_label, predict_y);
							if(predict_y != target_label)
								mze += 1;
							diff = fabs(predict_y -target_label);
							mae += diff;
							mse += diff*diff;
							// printf("%d %.3f %.3f\n",mze, mae, diff);
						}				
					stop=clock();
					T = (double)(stop-start)/CLOCKS_PER_SEC;
					printf("%.4f %.3f %d %.3f %.3f %.3f\n",T,v, nSV, mze/tl, mae/tl, mse/tl);

				}
			}



		}

		iter++;
		if(iter % 10 == 0)
		{
			info(".");
		}

		if(stopping < eps_shrink)
		{
			if(stopping < eps && start_from_all == true)
				break;
			else
			{
				active_size = l - tl;
				for(i=0;i<l;i++)
					active_size_i[i] = nr_class;
				// info("*");
				eps_shrink = max(eps_shrink/2, eps);
				start_from_all = true;
			}
		}
		else
			start_from_all = false;
		
	}
	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\n");

	// calculate objective value
	double v = 0;
	int nSV = 0;
	for(i=0;i<w_size*nr_class;i++)
		v += w[i]*w[i];
	v = 0.5*v;
	for(i=0;i<l*nr_class;i++)
	{
		v += alpha[i];
		if(fabs(alpha[i]) > 0)
			nSV++;
	}
	for(i=0;i<l;i++)
		v -= alpha[i*nr_class+(int)prob->y[i]];
	info("Objective value = %lf\n",v);
	info("nSV = %d\n",nSV);

	delete [] alpha;
	delete [] alpha_new;
	delete [] index;
	delete [] QD;
	delete [] d_ind;
	delete [] d_val;
	delete [] alpha_index;
	delete [] y_index;
	delete [] active_size_i;
}



// Crammer-Singer Ordinal Regression(CSOR)

class Solver_CSOR
{
	public:
		Solver_CSOR(const problem *prob, int nr_class, double C, double eps=0.1, int max_iter=200);
		~Solver_CSOR();
		void Solve(double *w);
	private:
		double *B, C, *G;
		int w_size, l;
		int nr_class;
		int max_iter;
		double eps;
		const problem *prob;
};

Solver_CSOR::Solver_CSOR(const problem *prob, int nr_class, double C, double eps, int max_iter)
{
	this->w_size = prob->n;
	this->l = prob->l;
	this->nr_class = nr_class;
	this->eps = eps;
	this->max_iter = max_iter;
	this->prob = prob;
	this->B = new double[nr_class];
	this->G = new double[nr_class];
	this->C = C;
}

Solver_CSOR::~Solver_CSOR()
{
	delete[] B;
	delete[] G;
}


void Solver_CSOR::Solve(double *w)
{





	int i, m, s;
	int iter = 0;
	double *alpha =  new double[l*nr_class];
	double *alpha_new = new double[nr_class-1];
	int *index = new int[l];
	double *QD = new double[l];
	int *alpha_index = new int[(nr_class-1)*l];
	double *b = new double[nr_class];
	// int *alphai_index = new int[nr_class*l];
	int *y_index = new int[l];
	int active_size = l;
	int *active_size_i = new int[l];
	// double eps_shrink = max(10.0*eps, 1.0); // stopping tolerance for shrinking
	// bool start_from_all = true;

	// Initial alpha can be set here. Note that 
	// sum_m alpha[i*nr_class+m] = 0, for all i=1,...,l-1
	// alpha[i*nr_class+m] <= C[GETI(i)] if prob->y[i] == m
	// alpha[i*nr_class+m] <= 0 if prob->y[i] != m
	// If initial alpha isn't zero, uncomment the for loop below to initialize w
	for(i=0;i<l*nr_class;i++)
		alpha[i] = 0;

	for(i=0;i<w_size*nr_class;i++)
		w[i] = 0;
	for(i=0;i<l;i++)
	{
		int count = 0;
		y_index[i] = (int)prob->y[i];
		for(m=0;m<nr_class;m++)
		{
			// alpha_index[i*nr_class+m] = m;
			if(m != y_index[i])
				{
					alpha_index[i*(nr_class-1)+count] = m;
					count++;
				}
		}
		

		feature_node *xi = prob->x[i];
		QD[i] = sparse_operator::nrm2_sq(xi);
		active_size_i[i] = nr_class-1;
		index[i] = i;
	}


    //shrinking
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration
    double violation=0;
	while(iter < max_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;				
		

		for(i=0;i<active_size;i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}

		for(s=0;s<active_size;s++)
		{
			violation = 0;
			i = index[s];
			int yi = y_index[i];
			double Ai = QD[i];
			double *alpha_i = &alpha[i*nr_class];
			int *alpha_index_i = &alpha_index[i*(nr_class-1)];
			int m;
			for(m=0;m<nr_class;m++)
				{
					B[m] = 0;
					b[m] = 0;
				}


			if(Ai > 0)
			{

				int Ui = active_size_i[i];				
				feature_node *xi = prob->x[i];
				while(xi->index!= -1)
				{
					double *w_i = &w[(xi->index-1)*nr_class];
					for(m=0;m<Ui;m++)
						{
							B[alpha_index_i[m]] += w_i[alpha_index_i[m]]*(xi->value);
						}
					B[yi] += w_i[yi]*(xi->value);
					xi++;
				}
								
				for(m=0;m<Ui;m++)
				{
					double yim = (yi>alpha_index_i[m])?1:-1;
					b[alpha_index_i[m]]= yim+B[alpha_index_i[m]]-B[yi];
					b[yi] += b[alpha_index_i[m]];
				}	
				double eta = b[yi]/(1+Ui);


				double PG;
				for(m=0;m<active_size_i[i];m++)
				{

					PG = -b[alpha_index_i[m]];
					if ((alpha_i[alpha_index_i[m]] == 0 && yi>alpha_index_i[m])||
					 (alpha_i[alpha_index_i[m]] == -C && yi<alpha_index_i[m]))
					{
						if (PG > Gmax_old)
						{
							active_size_i[i]--;						
							swap(alpha_index_i[m], alpha_index_i[active_size_i[i]]);
							m--;
							continue;
						}
						else if (PG < 0)
							violation = -PG;
					}
					else if ((alpha_i[alpha_index_i[m]] == C && yi>alpha_index_i[m])||
					 (alpha_i[alpha_index_i[m]] == 0 && yi<alpha_index_i[m]))
					{

						if (PG < -Gmax_old)
						{
							active_size_i[i]--;						
							swap(alpha_index_i[m], alpha_index_i[active_size_i[i]]);
							m--;
							continue;
						}
						else if (PG > 0)
							violation = PG;
					}
					else
						violation = fabs(PG);

					Gmax_new = max(Gmax_new, violation);
					Gnorm1_new += violation;
				}


				if(active_size_i[i] <= 0)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}

				//solve submodel
				double alpha_inew,alpha_iold;
				double *d = new double[active_size_i[i]];
				double h;
				for(m=0;m<active_size_i[i];m++)
				{
					alpha_iold = alpha_i[alpha_index_i[m]];
					h = (b[alpha_index_i[m]] - eta)/Ai;
					if(yi>alpha_index_i[m])
						alpha_inew = max(0.0,min(C,alpha_iold + h));
					else
						alpha_inew = max(-C,min(0.0,alpha_iold + h));
					alpha_i[alpha_index_i[m]] = alpha_inew;
					d[alpha_index_i[m]] = alpha_inew - alpha_iold;
				}

				xi = prob->x[i];
				while(xi->index != -1)
				{
					double *w_i = &w[(xi->index-1)*nr_class];
					for(m=0;m<active_size_i[i];m++)
					{
						w_i[alpha_index_i[m]] -= d[alpha_index_i[m]]*xi->value;
						w_i[yi] += d[alpha_index_i[m]]*xi->value;
					}
					xi++;
				}

			}
		}


		// // calculate objective value
		// double v = 0;
		// int nSV = 0;
		// for(i=0;i<w_size*nr_class;i++)
		// 	v += w[i]*w[i];
		// v = 0.5*v;
		// for(i=0;i<l;i++)
		// {
		// 	int yi = (int)prob->y[i];	
		// 	for(m=0;m<nr_class;m++)
		// 	{
		// 		if(yi < m)  
		// 		{
		// 			v += alpha[i*nr_class+m];
		// 		    if(fabs(alpha[i*nr_class+m]) > 0)
		// 			   nSV++;	
		// 		}
		// 		else if(yi>m)
		// 		{
		// 			v -= alpha[i*nr_class+m];
		// 		    if(fabs(alpha[i*nr_class+m]) > 0)
		// 			   nSV++;	
		// 		}
		// 	}
		// }
	
		// // info("Objective value = %lf\n",v);
		// info("Objective value = %lf nSV = %d  violation = %.3f Gnorm1_new = %.3f Gmax_new = %.3f\n",v,nSV, violation, Gnorm1_new,Gmax_new);	



		if(iter == 0)
			Gnorm1_init = Gnorm1_new;
		iter++;
		if(iter % 10 == 0)
			info(".");

		if(Gnorm1_new <= eps*Gnorm1_init)
		{
			if(active_size == l)
				break;
			else
			{
				active_size = l;
				info("*");
				Gmax_old = INF;
				continue;
			}
		}
		Gmax_old = Gmax_new;



		// ///measures vs time
		// {
		// 	int nr_w = nr_class;
		// 	int t;
		// 	double mze=0;
		// 	double mae = 0.0, mse = 0.0, T, diff,target_label,predict_y;	

		// 	for(int t=l;t<prob->l;t++)		
		// 		{
		// 			target_label = (double)prob->y[t];
		// 			predict_y = (double)predict_label(prob->x[t], w, prob->label, nr_w, w_size, nr_class);
		// 			// info("target_label=%.3f predict_y=%.3f\n",target_label, predict_y);
		// 			if(predict_y != target_label)
		// 				mze += 1;
		// 			diff = fabs(predict_y -target_label);
		// 			mae += diff;
		// 			mse += diff*diff;
		// 			// printf("%d %.3f %.3f\n",mze, mae, diff);
		// 		}				
		// 	stop=clock();
		// 	T = (double)(stop-start)/CLOCKS_PER_SEC;
		// 	printf("%.4f %.3f %.3f %.3f\n",T, mze/tl, mae/tl, mse/tl);

		// }




	}

	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\n");

	// calculate objective value
	double v = 0;
	int nSV = 0;
	for(i=0;i<w_size*nr_class;i++)
		v += w[i]*w[i];
	v = 0.5*v;
	for(i=0;i<l;i++)
	{
		int yi = (int)prob->y[i];	
		for(m=0;m<nr_class;m++)
		{
			if(yi < m)  
			{
				v += alpha[i*nr_class+m];
			    if(fabs(alpha[i*nr_class+m]) > 0)
				   nSV++;	
			}
			else if(yi>m)
			{
				v -= alpha[i*nr_class+m];
			    if(fabs(alpha[i*nr_class+m]) > 0)
				   nSV++;	
			}
		}
	}

	info("Objective value = %lf\n",v);
	info("nSV = %d\n",nSV);

	delete [] alpha;
	delete [] alpha_new;
	delete [] index;
	delete [] QD;
	delete [] alpha_index;
	delete [] y_index;
	delete [] active_size_i;

}



// Crammer-Singer Ordinal Regression(CSOR) with all considering

class Solver_CSOR_all
{
	public:
		Solver_CSOR_all(const problem *prob, int nr_class, double C, double eps=0.1, int max_iter=500);
		~Solver_CSOR_all();
		void Solve(double *w);
	private:
		double C, *G;
		int w_size, l;
		int nr_class;
		int max_iter;
		double eps;
		const problem *prob;
};

Solver_CSOR_all::Solver_CSOR_all(const problem *prob, int nr_class, double C, double eps, int max_iter)
{
	this->w_size = prob->n;
	this->l = prob->l;
	this->nr_class = nr_class;
	this->eps = eps;
	this->max_iter = max_iter;
	this->prob = prob;
	this->G = new double[nr_class];
	this->C = C;
}

Solver_CSOR_all::~Solver_CSOR_all()
{
	delete[] G;
}


void Solver_CSOR_all::Solve(double *w)
{
	int tl = 0;
	int active_size = l -tl;
	clock_t start, stop;
	start=clock();


	if(prob->WithTime)
	{
		tl = (int)prob->l/5;
		active_size = l -tl;
					
	}


	int i, m, s;
	int iter = 0;
	double *alpha =  new double[l*(nr_class-1)];
	double *alpha_new = new double[nr_class-1];
	int *index = new int[l];
	double *QD = new double[l];
	int *alpha_index = new int[(nr_class-1)*l];
	double *b = new double[nr_class];
	// int *alphai_index = new int[nr_class*l];
	int *y_index = new int[l];
	int *active_size_i = new int[l];
	double eps_shrink = max(10.0*eps, 1.0); // stopping tolerance for shrinking
	bool start_from_all = true;

	// Initial alpha can be set here. Note that 
	// sum_m alpha[i*nr_class+m] = 0, for all i=1,...,l-1
	// alpha[i*nr_class+m] <= C[GETI(i)] if prob->y[i] == m
	// alpha[i*nr_class+m] <= 0 if prob->y[i] != m
	// If initial alpha isn't zero, uncomment the for loop below to initialize w
	for(i=0;i<l*(nr_class-1);i++)
		alpha[i] = 0;

	for(i=0;i<w_size*nr_class;i++)
		w[i] = 0;
	for(i=0;i<l;i++)
	{
		int count = 0;
		y_index[i] = (int)prob->y[i];
		for(m=0;m<nr_class;m++)
		{
			// alpha_index[i*nr_class+m] = m;
			if(m != y_index[i])
				{
					alpha_index[i*(nr_class-1)+count] = m;
					count++;
				}
		}
	
		feature_node *xi = prob->x[i];
		QD[i] = sparse_operator::nrm2_sq(xi);
		active_size_i[i] = nr_class-1;
		index[i] = i;
	}


    //shrinking
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration
    double violation=0;
	while(iter < max_iter)
	{
		double stopping = -INF;
		Gmax_new = 0;
		Gnorm1_new = 0;				
		

		for(i=0;i<active_size;i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}

		for(s=0;s<active_size;s++)
		{

			violation = 0;
			i = index[s];
			int yi = y_index[i];
			double Ai = QD[i];
			double *alpha_i = &alpha[i*(nr_class-1)];
			int *alpha_index_i = &alpha_index[i*(nr_class-1)];
			int m;
			for(m=0;m<nr_class;m++)
				{
					G[m] = -1;
					b[m] = 0;
				}


			if(Ai > 0)
			{

				int Ui = active_size_i[i];				
				feature_node *xi = prob->x[i];
				while(xi->index!= -1)
				{
					double *w_i = &w[(xi->index-1)*nr_class];
					for(m=0;m<Ui;m++)
						{
							double yim = (yi>alpha_index_i[m])?1:-1;
							G[alpha_index_i[m]] += yim*(w_i[alpha_index_i[m]]
								+w_i[yi])*(xi->value);
						}
					xi++;
				}
				
				double eta = 0;				
				for(m=0;m<Ui;m++)
				{
					double yim = (yi>alpha_index_i[m])?1:-1;
					b[alpha_index_i[m]]= -G[alpha_index_i[m]];
					eta += yim*b[alpha_index_i[m]];
				}	
				eta = eta/(1+Ui);


				double PG;
				for(m=0;m<active_size_i[i];m++)
				{

					PG = G[alpha_index_i[m]];
					if (alpha_i[alpha_index_i[m]] == 0)
					{
						if (PG > 0)
						{
							active_size_i[i]--;						
							swap(alpha_index_i[m], alpha_index_i[active_size_i[i]]);
							m--;
							continue;
						}
						else if (PG < 0)
							violation = -PG;
					}
					else if (alpha_i[alpha_index_i[m]] == C)
					{

						if (PG < 0)
						{
							active_size_i[i]--;						
							swap(alpha_index_i[m], alpha_index_i[active_size_i[i]]);
							m--;
							continue;
						}
						else if (PG > 0)
							violation = PG;
					}
					else
						violation = fabs(PG);

					Gmax_new = max(Gmax_new, violation);
					Gnorm1_new += violation;
				}


				if(active_size_i[i] <= 0)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}

				if(Gmax_new <= 0.1)
					continue;
				else
					stopping = max(Gmax_new, stopping);				

				//solve submodel
				double alpha_inew,alpha_iold;
				double *d = new double[active_size_i[i]];
				double h;
				for(m=0;m<active_size_i[i];m++)
				{
					double yim = (yi>alpha_index_i[m])?1:-1;
					alpha_iold = alpha_i[alpha_index_i[m]];
					h = (b[alpha_index_i[m]] - eta*yim)/Ai;
					alpha_inew = max(0.0,min(C,alpha_iold + h));
					alpha_i[alpha_index_i[m]] = alpha_inew;
					d[m] = yim*(alpha_inew - alpha_iold);
				}

				xi = prob->x[i];
				while(xi->index != -1)
				{
					double *w_i = &w[(xi->index-1)*nr_class];
					for(m=0;m<active_size_i[i];m++)
					{
						w_i[alpha_index_i[m]] += d[m]*xi->value;
						w_i[yi] += d[m]*xi->value;
					}
					xi++;
				}

			}



			if(prob->WithTime)
			{
				if(s%(active_size/5)==0)
				///measures vs time
				{

					// calculate objective value
					double v = 0;
					int nSV = 0;
					for(i=0;i<w_size*nr_class;i++)
						v += w[i]*w[i];
					v = 0.5*v;
					for(i=0;i<l;i++)
					{
						for(m=0;m<nr_class-1;m++)
						{
							v -= alpha[i*(nr_class-1)+m];
							if(fabs(alpha[i*(nr_class-1)+m]) > 0)
								nSV++;	
						}
					}


					int nr_w = nr_class;
					// int t;
					double mze=0;
					double mae = 0.0, mse = 0.0, T, diff,target_label,predict_y;	

					for(int t=l-tl;t<prob->l;t++)		
						{
							target_label = (double)prob->y[t]+1;
							predict_y = (double)predict_label(prob->x[t], w, prob->label, nr_w, w_size, nr_class);
						    // info("target_label=%.3f predict_y=%.3f\n",target_label, predict_y);
							if(predict_y != target_label)
								mze += 1;
							diff = fabs(predict_y -target_label);
							mae += diff;
							mse += diff*diff;
							// printf("%d %.3f %.3f\n",mze, mae, diff);
						}				
					stop=clock();
					T = (double)(stop-start)/CLOCKS_PER_SEC;
					printf("%.4f %.lf %d %.4f %.4f %.4f\n",T, v, nSV, mze/tl, mae/tl, mse/tl);

				}				
			}
		}		
	
		// // info("Objective value = %lf\n",v);
		// info("Objective value = %lf nSV = %d  violation = %.3f Gnorm1_new = %.3f Gmax_new = %.3f active_size=%d\n",
		// 	v,nSV, violation, Gnorm1_new,Gmax_new,active_size);	



		if(iter == 0)
			Gnorm1_init = Gnorm1_new;
		iter++;
		// if(iter % 10 == 0)
		// 	info(".");

		// if(Gnorm1_new <= eps*Gnorm1_init )
		// {
		// 	if(active_size == l )
		// 		break;
		// 	else
		// 	{
		// 		active_size = l;
		// 		for(i=0;i<l;i++)
		// 			active_size_i[i] = nr_class-1;				
		// 		info("*");
		// 		Gmax_old = INF;
		// 		continue;
		// 	}
		// }
		// Gmax_old = Gmax_new;


		if(stopping < eps_shrink)
		{
			if(Gnorm1_new <= eps*Gnorm1_init && start_from_all == true)
				break;
			else
			{
				active_size = l-tl;
				for(i=0;i<l;i++)
					active_size_i[i] = nr_class-1;				
				// info("*");
				eps_shrink = max(eps_shrink/2, eps);
				start_from_all = true;
				Gmax_old = INF;
				// continue;
			}
		}
		else
			start_from_all = false;

		Gmax_old = Gmax_new;

	}

	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\n");

	// calculate objective value
	double v = 0;
	int nSV = 0;
	for(i=0;i<w_size*nr_class;i++)
		v += w[i]*w[i];
	v = 0.5*v;
	for(i=0;i<l;i++)
	{
		for(m=0;m<nr_class-1;m++)
		{
			v -= alpha[i*(nr_class-1)+m];
			if(fabs(alpha[i*(nr_class-1)+m]) > 0)
				nSV++;	
		}
	}

	info("Objective value = %lf\n",v);
	info("nSV = %d\n",nSV);

	delete [] alpha;
	delete [] alpha_new;
	delete [] index;
	delete [] QD;
	delete [] alpha_index;
	delete [] y_index;
	delete [] active_size_i;

}




// Crammer-Singer Ordinal Regression(CSOR)

class Solver_SSVOR
{
	public:
		Solver_SSVOR(const problem *prob, int nr_class, double C, double eps=0.1, int max_iter=20);
		~Solver_SSVOR();
		void Solve(double *w);
	private:
		void solve_sub_problem(double A_i, int active_i, double *Yi, double *d, double *G);
		double C, *G;
		int w_size, l;
		int nr_class;
		int max_iter;
		double eps;
		const problem *prob;
};

Solver_SSVOR::Solver_SSVOR(const problem *prob, int nr_class, double C, double eps, int max_iter)
{
	this->w_size = prob->n;
	this->l = prob->l;
	this->nr_class = nr_class;
	this->eps = eps;
	this->max_iter = max_iter;
	this->prob = prob;
	this->G = new double[nr_class-1];
	this->C = C;
}

Solver_SSVOR::~Solver_SSVOR()
{
	delete[] G;
}

void Solver_SSVOR::solve_sub_problem(double A_i, int active_i, double *Yi, double *d, double *G)
{
	int r;
	double *b = new double[active_i];
	double t;
	b[0] = -Yi[0]*G[0];
	for(r=1;r<active_i;r++)
	{
		t = (double)r;
		// t = t/(t+1);
		b[r] = -Yi[r]*G[r] - t/(t+1)*b[r-1];
	}
	
	for(r=active_i-1;r>=0;r--)
	{
		t = (double)r;
		t = (t+1)/(t+2)/A_i;
		if(r == active_i-1)
			d[r] = t*b[r];
		else
			d[r] = t*(b[r] -d[r+1]);
	}
}

///new
void Solver_SSVOR::Solve(double *w)
{
	int tl = 0;
	int active_size = l -tl;
	clock_t start, stop;
	start=clock();


	if(prob->WithTime)
	{
		tl = (int)prob->l*3/10;
		active_size = l -tl;
					
	}	


	int i, m, s;
	int iter = 0;
	double *alpha =  new double[l*(nr_class-1)];
	double *h = new double[nr_class-1];
	int *index = new int[l];
	double *QD = new double[l];
	double *y_index = new double[l];

	for(i=0;i<l*(nr_class-1);i++)
		alpha[i] = 0;

	for(i=0;i<w_size*nr_class;i++)
		w[i] = 0;
	for(i=0;i<l;i++)
	{
		y_index[i] = (double)prob->y[i];	
		feature_node *xi = prob->x[i];
		QD[i] = sparse_operator::nrm2_sq(xi);
		index[i] = i;
	}


    //shrinking
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration
	while(iter < max_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;				
		double violation=0;


		for(i=0;i<active_size;i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}

		for(s=0;s<active_size;s++)
		{
			
			i = index[s];
			double yi = y_index[i];
			double Ai = QD[i];
			double *alpha_i = &alpha[i*(nr_class-1)];
			
			if(Ai > 0)
			{
				double *Yi = new double[nr_class-1];
				for(m=0;m<nr_class-1;m++)
					{
						Yi[m] = (yi>m)?1:-1;
						G[m] = -1;
					}
				feature_node *xi = prob->x[i];
				while(xi->index!= -1)
				{
					double *w_i = &w[(xi->index-1)*nr_class];
					for(m=0;m<nr_class-1;m++)
						{
							G[m] += Yi[m]*(w_i[m]+w_i[m+1])*(xi->value);
						}
					xi++;
				}				

				double violation_i=0;
				for(m=0;m<nr_class-1;m++)
				{
					violation = 0;
					if (alpha_i[m] == 0)
					{						
						violation = max(0.0, -G[m]);
					}
					else if (alpha_i[m] == C)
					{

						violation = max(0.0, G[m]);
					}
					else
						violation = fabs(G[m]);

					Gmax_new += max(Gmax_new, violation);
					Gnorm1_new += violation;
					violation_i = max(violation_i,violation);
				}
					

				if(fabs(violation_i) < 0.1)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}

				//solve submodel				
				solve_sub_problem(Ai, nr_class-1,Yi,h, G);


				double alpha_inew,alpha_iold;
				double *d = new double[nr_class-1];
				for(m=0;m<nr_class-1;m++)
				{
					alpha_iold = alpha_i[m];					
					alpha_inew = max(0.0,min(C,alpha_iold + Yi[m]*h[m]));
					alpha_i[m] = alpha_inew;
					d[m] = Yi[m]*(alpha_inew - alpha_iold);

				}

				xi = prob->x[i];
				while(xi->index != -1)
				{
					double *w_i = &w[(xi->index-1)*nr_class];
					for(m=0;m<nr_class-1;m++)
					{
						w_i[m] += d[m]*xi->value;
						w_i[m+1] += d[m]*xi->value;
					}
					xi++;
				}


			}

			if(prob->WithTime)
			{			
				if(s%(active_size/5)==0)
				///measures vs time
				{
					// calculate objective value
					double v = 0;
					int nSV = 0;
					for(i=0; i<w_size*nr_class; i++)
						v += w[i]*w[i];
					v = v/2;
					for(i=0; i<l*(nr_class-1); i++)
					{
						v -= alpha[i];
						if(alpha[i] > 0)
							++nSV;
					}
					
					// // info("Objective value = %lf\n",v);
					// info("Objective value = %lf nSV = %d  violation = %.3f Gnorm1_new = %.3f Gmax_new = %.3f\n",v,nSV, violation, Gnorm1_new,Gmax_new);	


					int nr_w = nr_class;
					// int t;
					double mze=0;
					double mae = 0.0, mse = 0.0, T, diff,target_label,predict_y;	

					for(int t=l -tl;t<prob->l;t++)		
						{
							target_label = (double)prob->y[t]+1;
							predict_y = (double)predict_label(prob->x[t], w, prob->label, nr_w, w_size, nr_class);
							// info("target_label=%.3f predict_y=%.3f\n",target_label, predict_y);
							if(predict_y != target_label)
								mze += 1;
							diff = fabs(predict_y -target_label);
							mae += diff;
							mse += diff*diff;
							// printf("%d %.3f %.3f\n",mze, mae, diff);
						}				
					stop=clock();
					T = (double)(stop-start)/CLOCKS_PER_SEC;
					printf("%.4f %.3f %.3f %.3f %.3f\n",T,v, mze/tl, mae/tl, mse/tl);

				}
			}

		}


		if(iter == 0)
			// Gnorm1_init = Gnorm1_new;
		    {
		    	Gnorm1_new = (nr_class-1)*l;
		    	Gnorm1_init = Gnorm1_new;
		    }
		iter++;
		if(iter % 10 == 0)
			info(".");

		if(Gnorm1_new <= eps*Gnorm1_init)
		{
			if(active_size == l-tl)
				break;
			else
			{
				active_size = l-tl;				
				info("*");
				Gmax_old = INF;
				continue;
			}
		}
		Gmax_old = Gmax_new;

	}

	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\n");

	// calculate objective value
	double v = 0;
	int nSV = 0;
	for(i=0; i<w_size*nr_class; i++)
		v += w[i]*w[i];
	v = v/2;
	for(i=0; i<l*(nr_class-1); i++)
	{
		v -= alpha[i];
		if(alpha[i] > 0)
			++nSV;
	}

	info("Objective value = %lf\n",v);
	info("nSV = %d\n",nSV);

	delete [] alpha;
	delete [] h;
	delete [] index;
	delete [] QD;
	delete [] y_index;

}


/*
void Solver_SSVOR::Solve(double *w)
{
	int tl = 0;
	int active_size = l -tl;
	clock_t start, stop;
	start=clock();


	if(prob->WithTime)
	{
		tl = (int)prob->l*3/10;
		active_size = l -tl;
					
	}	


	int i, m, s;
	int iter = 0;
	double *alpha =  new double[l*(nr_class-1)];
	double *h = new double[nr_class-1];
	int *index = new int[l];
	double *QD = new double[l];
	int *alpha_index = new int[(nr_class-1)*l];
	// int *alphai_index = new int[nr_class*l];
	double *y_index = new double[l];
	int *active_size_i = new int[l];
	// double eps_shrink = max(10.0*eps, 1.0); // stopping tolerance for shrinking
	// bool start_from_all = true;

	for(i=0;i<l*(nr_class-1);i++)
		alpha[i] = 0;

	for(i=0;i<w_size*nr_class;i++)
		w[i] = 0;
	for(i=0;i<l;i++)
	{
		y_index[i] = (double)prob->y[i];
		for(m=0;m<nr_class-1;m++)
			alpha_index[i*(nr_class-1)+m] = m;
		
		feature_node *xi = prob->x[i];
		QD[i] = sparse_operator::nrm2_sq(xi);
		active_size_i[i] = nr_class-1;
		index[i] = i;
	}


    //shrinking
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration
	while(iter < max_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;				
		double violation=0;


		for(i=0;i<active_size;i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}

		for(s=0;s<active_size;s++)
		{
			
			i = index[s];
			double yi = y_index[i];
			double Ai = QD[i];
			double *alpha_i = &alpha[i*(nr_class-1)];
			int *alpha_index_i = &alpha_index[i*(nr_class-1)];
			
			if(Ai > 0)
			{
				double *Yi = new double[active_size_i[i]];
				for(m=0;m<active_size_i[i];m++)
					{
						Yi[alpha_index_i[m]] = (yi>alpha_index_i[m])?1:-1;
						G[alpha_index_i[m]] = -1;
					}
				feature_node *xi = prob->x[i];
				while(xi->index!= -1)
				{
					double *w_i = &w[(xi->index-1)*nr_class];
					for(m=0;m<active_size_i[i];m++)
						{
							G[alpha_index_i[m]] += Yi[alpha_index_i[m]]*(w_i[alpha_index_i[m]]
								+w_i[alpha_index_i[m]+1])*(xi->value);
						}
					xi++;
				}				

				double violation_i=0;
				for(m=0;m<active_size_i[i];m++)
				{
					violation = 0;
					if (alpha_i[alpha_index_i[m]] == 0)
					{						
						violation = max(0.0, -G[alpha_index_i[m]]);
					}
					else if (alpha_i[alpha_index_i[m]] == C)
					{

						violation = max(0.0, G[alpha_index_i[m]]);
					}
					else
						violation = fabs(G[alpha_index_i[m]]);

					Gmax_new += max(Gmax_new, violation);
					Gnorm1_new += violation;
					violation_i = max(violation_i,violation);
				}
					

				if(fabs(violation_i) < 0.1)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}

				//solve submodel				
				solve_sub_problem(Ai, active_size_i[i],Yi,h, G);


				double alpha_inew,alpha_iold;
				double *d = new double[active_size_i[i]];
				for(m=0;m<active_size_i[i];m++)
				{
					alpha_iold = alpha_i[alpha_index_i[m]];					
					alpha_inew = max(0.0,min(C,alpha_iold + Yi[alpha_index_i[m]]*h[alpha_index_i[m]]));
					alpha_i[alpha_index_i[m]] = alpha_inew;
					d[alpha_index_i[m]] = Yi[alpha_index_i[m]]*(alpha_inew - alpha_iold);

				}

				xi = prob->x[i];
				while(xi->index != -1)
				{
					double *w_i = &w[(xi->index-1)*nr_class];
					for(m=0;m<active_size_i[i];m++)
					{
						w_i[alpha_index_i[m]] += d[alpha_index_i[m]]*xi->value;
						w_i[alpha_index_i[m]+1] += d[alpha_index_i[m]]*xi->value;
					}
					xi++;
				}


			}

			if(prob->WithTime)
			{			
				if(s%(active_size/5)==0)
				///measures vs time
				{
					// calculate objective value
					double v = 0;
					int nSV = 0;
					for(i=0; i<w_size*nr_class; i++)
						v += w[i]*w[i];
					v = v/2;
					for(i=0; i<l*(nr_class-1); i++)
					{
						v -= alpha[i];
						if(alpha[i] > 0)
							++nSV;
					}
					
					// // info("Objective value = %lf\n",v);
					// info("Objective value = %lf nSV = %d  violation = %.3f Gnorm1_new = %.3f Gmax_new = %.3f\n",v,nSV, violation, Gnorm1_new,Gmax_new);	


					int nr_w = nr_class;
					// int t;
					double mze=0;
					double mae = 0.0, mse = 0.0, T, diff,target_label,predict_y;	

					for(int t=l -tl;t<prob->l;t++)		
						{
							target_label = (double)prob->y[t]+1;
							predict_y = (double)predict_label(prob->x[t], w, prob->label, nr_w, w_size, nr_class);
							// info("target_label=%.3f predict_y=%.3f\n",target_label, predict_y);
							if(predict_y != target_label)
								mze += 1;
							diff = fabs(predict_y -target_label);
							mae += diff;
							mse += diff*diff;
							// printf("%d %.3f %.3f\n",mze, mae, diff);
						}				
					stop=clock();
					T = (double)(stop-start)/CLOCKS_PER_SEC;
					printf("%.4f %.3f %.3f %.3f %.3f\n",T,v, mze/tl, mae/tl, mse/tl);

				}
			}

		}


		if(iter == 0)
			// Gnorm1_init = Gnorm1_new;
		    {
		    	Gnorm1_new = (nr_class-1)*l;
		    	Gnorm1_init = Gnorm1_new;
		    }
		iter++;
		// if(iter % 10 == 0)
		// 	info(".");

		if(Gnorm1_new <= eps*Gnorm1_init)
		{
			if(active_size == l-tl)
				break;
			else
			{
				active_size = l-tl;				
				// info("*");
				Gmax_old = INF;
				continue;
			}
		}
		Gmax_old = Gmax_new;

	}

	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\n");

	// calculate objective value
	double v = 0;
	int nSV = 0;
	for(i=0; i<w_size*nr_class; i++)
		v += w[i]*w[i];
	v = v/2;
	for(i=0; i<l*(nr_class-1); i++)
	{
		v -= alpha[i];
		if(alpha[i] > 0)
			++nSV;
	}

	info("Objective value = %lf\n",v);
	info("nSV = %d\n",nSV);

	delete [] alpha;
	delete [] h;
	delete [] index;
	delete [] QD;
	delete [] alpha_index;
	delete [] y_index;
	delete [] active_size_i;

}
*/

// Crammer-Singer Ordinal Regression(CSOR) 

class Solver_NPSSVOR
{
	public:
		Solver_NPSSVOR(const problem *prob, int nr_class, double C1, double C2, double p, double eps=0.1, int max_iter=20);
		~Solver_NPSSVOR();
		void Solve(double *w);
	private:
		void solve_sub_problem(double A_i, int active_i, double *Yi, double *d, double *G);
		double C1,C2,p, *G;
		int w_size, l;
		int nr_class;
		int max_iter;
		double eps;
		const problem *prob;
};

Solver_NPSSVOR::Solver_NPSSVOR(const problem *prob, int nr_class, double C1, double C2, double p, double eps, int max_iter)
{
	this->w_size = prob->n;
	this->l = prob->l;
	this->nr_class = nr_class;
	this->eps = eps;
	this->max_iter = max_iter;
	this->prob = prob;
	this->G = new double[nr_class];
	this->C1 = C1;
	this->C2 = C2;
	this->p = p;		
}

Solver_NPSSVOR::~Solver_NPSSVOR()
{
	delete[] G;
}

void Solver_NPSSVOR::solve_sub_problem(double A_i, int active_i, double *Yi, double *d, double *G)
{
	int r;
	double *b = new double[active_i];
	double t;
	b[0] = -Yi[0]*G[0];
	for(r=1;r<active_i;r++)
	{
		t = (double)r;
		t = t/(t+1);
		b[r] = -Yi[r]*G[r] - t*b[r-1];
	}
	
	for(r=active_i-1;r>=0;r--)
	{
		t = (double)r;
		t = (t+1)/(t+2);
		t=t/A_i;
		if(r == active_i-1)
			d[r] = t*b[r];
		else
			d[r] = t*(b[r] -d[r+1]);
	}
}

///new
void Solver_NPSSVOR::Solve(double *w)
{
	int tl = 0;
	int active_size = l -tl;
	clock_t start, stop;
	start=clock();


	if(prob->WithTime)
	{
		tl = (int)prob->l*3/10;
		active_size = l -tl;
					
	}	


	int i, m, s;
	int iter = 0;
	double *alpha =  new double[l*nr_class];
	double *h = new double[nr_class];
	int *index = new int[l];
	double *QD = new double[l];
	// int *alphai_index = new int[nr_class*l];
	int  *y_index = new int[l];
	// double eps_shrink = max(10.0*eps, 1.0); // stopping tolerance for shrinking
	// bool start_from_all = true;

	for(i=0;i<l*nr_class;i++)
		alpha[i] = 0;

	for(i=0;i<w_size*nr_class;i++)
		w[i] = 0;
	for(i=0;i<l;i++)
	{
		y_index[i] = (int)prob->y[i];
		feature_node *xi = prob->x[i];
		QD[i] = sparse_operator::nrm2_sq(xi);
		index[i] = i;
	}


    //shrinking
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration
	while(iter < max_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;				
		double violation=0;

		for(i=0;i<active_size;i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}

		for(s=0;s<active_size;s++)
		{
			
			i = index[s];
			int yi = y_index[i];
			double Ai = QD[i];
			double *alpha_i = &alpha[i*nr_class];
			
			if(Ai > 0)
			{
				double *Yi = new double[nr_class];
				for(m=0;m<nr_class;m++)
					{
						Yi[m] = (yi>m)?1:-1;
						G[m] = (m<nr_class-1)?-1:0;
					}

				feature_node *xi = prob->x[i];
				while(xi->index!= -1)
				{
					double *w_i = &w[(xi->index-1)*nr_class];
					for(m=0;m<nr_class-1;m++)
						{
							G[m] += Yi[m]*(w_i[m]+w_i[m+1])*(xi->value);
						}									
					xi++;
				}


				double violation_i=0;
				for(m=0;m<nr_class-1;m++)
				{
					violation = 0;
					if (alpha_i[m] == 0)
					{						
						violation = max(0.0, -G[m]);
					}
					else if (alpha_i[m] == C2)
					{

						violation = max(0.0, G[m]);
					}
					else
						violation = fabs(G[m]);				

					Gmax_new += max(Gmax_new, violation);
					Gnorm1_new += violation;
					violation_i = max(violation_i,violation);		
				}

				//solve submodel				
				solve_sub_problem(Ai, nr_class-1,Yi, h, G);

				double alpha_inew,alpha_iold;
				double *d = new double[nr_class];
				for(m=0;m<nr_class-1;m++)
				{
					alpha_iold = alpha_i[m];									
					alpha_inew = max(0.0,min(C2, alpha_iold + Yi[m]*h[m]));
					alpha_i[m] = alpha_inew;
					d[m] = Yi[m]*(alpha_inew - alpha_iold);		
				}

				if(fabs(violation_i) > 1.0e-12)
				{
					xi = prob->x[i];
					while(xi->index != -1)
					{
						double *w_i = &w[(xi->index-1)*nr_class];
						for(m=0;m<nr_class-1;m++)
						{
							w_i[m] += d[m]*xi->value;
							w_i[m+1] += d[m]*xi->value;							
						}					
						xi++;
					}
				}

				{
					double Gp = G[nr_class-1] + p;
					double Gn = G[nr_class-1] - p;


					if(alpha_i[nr_class-1] == 0)
					{
						if(Gp < 0)
							violation = -Gp;
						else if(Gn > 0)
							violation = Gn;
					}
					else if(alpha_i[nr_class-1] >= C1)
					{
						if(Gp > 0)
							violation = Gp;
					}
					else if(alpha_i[nr_class-1] <= -C1)
					{
						if(Gn < 0)
							violation = -Gn;
					}
					else if(alpha_i[nr_class-1] > 0)
						violation = fabs(Gp);
					else
						violation = fabs(Gn);
					Gmax_new += max(Gmax_new, violation);
					Gnorm1_new += violation;
					violation_i = max(violation_i,violation);		


					if(fabs(violation_i) < 1.0e-12)
					{
						active_size--;
						swap(index[s], index[active_size]);
						s--;
						continue;
					}


					if(Gp < Ai*alpha_i[nr_class-1])
						h[nr_class-1] = -Gp/Ai;
					else if(Gn > Ai*alpha_i[nr_class-1])
						h[nr_class-1] = -Gn/Ai;
					else
						h[nr_class-1] = -alpha_i[nr_class-1];


					for(m=nr_class-1;m<nr_class;m++)
					{
						alpha_iold = alpha_i[m];				
						alpha_inew = max(-C1,min(C1, alpha_iold + h[m]));
						alpha_i[m] = alpha_inew;
						d[m] = Yi[m]*(alpha_inew - alpha_iold);		
					}
					while(xi->index!= -1)
					{
						double *w_i = &w[(xi->index-1)*nr_class];
						G[nr_class-1] += -w_i[yi]*(xi->value);									
						xi++;
					}	
				}

			}

			if(prob->WithTime)
			{			
				if(s%(active_size/5)==0)
				///measures vs time
				{
					// calculate objective value
					double v = 0;
					int nSV = 0;
					for(i=0; i<w_size*nr_class; i++)
						v += w[i]*w[i];
					v = v/2;
					for(i=0; i<nr_class*l; i++)
					{
						if(i%nr_class==nr_class-1)
							v +=p*fabs(alpha[i]);
						else
							v -= alpha[i];		
						if(fabs(alpha[i]) > 0)
							++nSV;
					}
					
					// // info("Objective value = %lf\n",v);
					// info("Objective value = %lf nSV = %d  violation = %.3f Gnorm1_new = %.3f Gmax_new = %.3f\n",v,nSV, violation, Gnorm1_new,Gmax_new);	


					int nr_w = nr_class;
					// int t;
					double mze=0;
					double mae = 0.0, mse = 0.0, T, diff,target_label,predict_y;	

					for(int t=l -tl;t<prob->l;t++)		
						{
							target_label = (double)prob->y[t]+1;
							predict_y = (double)predict_label(prob->x[t], w, prob->label, nr_w, w_size, nr_class);
							// info("target_label=%.3f predict_y=%.3f\n",target_label, predict_y);
							if(predict_y != target_label)
								mze += 1;
							diff = fabs(predict_y -target_label);
							mae += diff;
							mse += diff*diff;
							// printf("%d %.3f %.3f\n",mze, mae, diff);
						}				
					stop=clock();
					T = (double)(stop-start)/CLOCKS_PER_SEC;
					printf("%.4f %.3f %.3f %.3f %.3f\n",T,v, mze/tl, mae/tl, mse/tl);

				}
			}

		}


		if(iter == 0)
		    Gnorm1_init = Gnorm1_new;
		iter++;
		// if(iter % 10 == 0)
		// 	info(".");

		if(Gnorm1_new <= eps*Gnorm1_init)
		{
			if(active_size == l-tl)
				break;
			else
			{
				active_size = l-tl;				
				// info("*");
				Gmax_old = INF;
				continue;
			}
		}
		Gmax_old = Gmax_new;

	}
	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\n");

	// calculate objective value
	double v = 0;
	int nSV = 0;
	for(i=0; i<w_size*nr_class; i++)
		v += w[i]*w[i];
	v = v/2;
	for(i=0; i<l*nr_class; i++)
	{
		if(i%nr_class==nr_class-1)
			v +=p*fabs(alpha[i]);
		else
			v -= alpha[i];		
		if(fabs(alpha[i]) > 0)
			++nSV;
	}

	info("Objective value = %lf\n",v);
	info("nSV = %d\n",nSV);

	delete [] alpha;
	delete [] h;
	delete [] index;
	delete [] QD;
	delete [] y_index;

}



/*

void Solver_NPSSVOR::Solve(double *w)
{
	int tl = 0;
	int active_size = l -tl;
	clock_t start, stop;
	start=clock();


	if(prob->WithTime)
	{
		tl = (int)prob->l*3/10;
		active_size = l -tl;
					
	}	


	int i, m, s;
	int iter = 0;
	double *alpha =  new double[l*nr_class];
	double *h = new double[nr_class];
	int *index = new int[l];
	double *QD = new double[l];
	// int *alphai_index = new int[nr_class*l];
	int  *y_index = new int[l];
	// double eps_shrink = max(10.0*eps, 1.0); // stopping tolerance for shrinking
	// bool start_from_all = true;

	for(i=0;i<l*nr_class;i++)
		alpha[i] = 0;

	for(i=0;i<w_size*nr_class;i++)
		w[i] = 0;
	for(i=0;i<l;i++)
	{
		y_index[i] = (int)prob->y[i];
		feature_node *xi = prob->x[i];
		QD[i] = sparse_operator::nrm2_sq(xi);
		index[i] = i;
	}


    //shrinking
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration
	while(iter < max_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;				
		double violation=0;

		for(i=0;i<active_size;i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}

		for(s=0;s<active_size;s++)
		{
			
			i = index[s];
			int yi = y_index[i];
			double Ai = QD[i];
			double *alpha_i = &alpha[i*nr_class];
			
			if(Ai > 0)
			{
				double *Yi = new double[nr_class];
				for(m=0;m<nr_class;m++)
					{
						Yi[m] = (yi>m)?1:-1;
						G[m] = (m<nr_class-1)?-1:0;
					}

				feature_node *xi = prob->x[i];
				while(xi->index!= -1)
				{
					double *w_i = &w[(xi->index-1)*nr_class];
					for(m=0;m<nr_class;m++)
						{
							if(m<nr_class-1)
								G[m] += Yi[m]*(w_i[m]+w_i[m+1])*(xi->value);
							else
								G[m] += Yi[m]*w_i[yi]*(xi->value);	
						}									
					xi++;
				}

				double Gp = G[nr_class-1] + p;
				double Gn = G[nr_class-1] - p;


				double violation_i=0;
				for(m=0;m<nr_class;m++)
				{
					if(m<nr_class-1)
					{
						violation = 0;
						if (alpha_i[m] == 0)
						{						
							violation = max(0.0, -G[m]);
						}
						else if (alpha_i[m] == C2)
						{

							violation = max(0.0, G[m]);
						}
						else
							violation = fabs(G[m]);				
					}
					else
					{
						if(alpha_i[m] == 0)
						{
							if(Gp < 0)
								violation = -Gp;
							else if(Gn > 0)
								violation = Gn;
						}
						else if(alpha_i[m] >= C1)
						{
							if(Gp > 0)
								violation = Gp;
						}
						else if(alpha_i[m] <= -C1)
						{
							if(Gn < 0)
								violation = -Gn;
						}
						else if(alpha_i[m] > 0)
							violation = fabs(Gp);
						else
							violation = fabs(Gn);
					}
					Gmax_new += max(Gmax_new, violation);
					Gnorm1_new += violation;
					violation_i = max(violation_i,violation);		
				}

				
				if(fabs(violation_i) < 1.0e-12)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}

				//solve submodel				
				solve_sub_problem(Ai, nr_class-1,Yi, h, G);










				if(Gp < Ai*alpha_i[nr_class-1])
					h[nr_class-1] = -Gp/Ai;
				else if(Gn > Ai*alpha_i[nr_class-1])
					h[nr_class-1] = -Gn/Ai;
				else
					h[nr_class-1] = -alpha_i[nr_class-1];


				double alpha_inew,alpha_iold;
				double *d = new double[nr_class];
				for(m=0;m<nr_class;m++)
				{
					alpha_iold = alpha_i[m];	
					if(m<nr_class-1)									
						alpha_inew = max(0.0,min(C2, alpha_iold + Yi[m]*h[m]));
					else				
						alpha_inew = max(-C1,min(C1, alpha_iold + h[m]));
					alpha_i[m] = alpha_inew;
					d[m] = Yi[m]*(alpha_inew - alpha_iold);		
				}

				if(fabs(violation_i) > 1.0e-12)
				{
					xi = prob->x[i];
					while(xi->index != -1)
					{
						double *w_i = &w[(xi->index-1)*nr_class];
						for(m=0;m<nr_class;m++)
						{
							if(m<nr_class-1)
							{
								w_i[m] += d[m]*xi->value;
								w_i[m+1] += d[m]*xi->value;							
							}
							else
							{
								w_i[yi] += d[m]*xi->value;
							}
						}					
						xi++;
					}
				}


			}

			if(prob->WithTime)
			{			
				if(s%(active_size/5)==0)
				///measures vs time
				{
					// calculate objective value
					double v = 0;
					int nSV = 0;
					for(i=0; i<w_size*nr_class; i++)
						v += w[i]*w[i];
					v = v/2;
					for(i=0; i<nr_class*l; i++)
					{
						if(i%nr_class==nr_class-1)
							v +=p*fabs(alpha[i]);
						else
							v -= alpha[i];		
						if(fabs(alpha[i]) > 0)
							++nSV;
					}
					
					// // info("Objective value = %lf\n",v);
					// info("Objective value = %lf nSV = %d  violation = %.3f Gnorm1_new = %.3f Gmax_new = %.3f\n",v,nSV, violation, Gnorm1_new,Gmax_new);	


					int nr_w = nr_class;
					// int t;
					double mze=0;
					double mae = 0.0, mse = 0.0, T, diff,target_label,predict_y;	

					for(int t=l -tl;t<prob->l;t++)		
						{
							target_label = (double)prob->y[t]+1;
							predict_y = (double)predict_label(prob->x[t], w, prob->label, nr_w, w_size, nr_class);
							// info("target_label=%.3f predict_y=%.3f\n",target_label, predict_y);
							if(predict_y != target_label)
								mze += 1;
							diff = fabs(predict_y -target_label);
							mae += diff;
							mse += diff*diff;
							// printf("%d %.3f %.3f\n",mze, mae, diff);
						}				
					stop=clock();
					T = (double)(stop-start)/CLOCKS_PER_SEC;
					printf("%.4f %.3f %.3f %.3f %.3f\n",T,v, mze/tl, mae/tl, mse/tl);

				}
			}

		}


		if(iter == 0)
		    Gnorm1_init = Gnorm1_new;
		iter++;
		// if(iter % 10 == 0)
		// 	info(".");

		if(Gnorm1_new <= eps*Gnorm1_init)
		{
			if(active_size == l-tl)
				break;
			else
			{
				active_size = l-tl;				
				// info("*");
				Gmax_old = INF;
				continue;
			}
		}
		Gmax_old = Gmax_new;

	}

	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\n");

	// calculate objective value
	double v = 0;
	int nSV = 0;
	for(i=0; i<w_size*nr_class; i++)
		v += w[i]*w[i];
	v = v/2;
	for(i=0; i<l*nr_class; i++)
	{
		if(i%nr_class==nr_class-1)
			v +=p*fabs(alpha[i]);
		else
			v -= alpha[i];		
		if(fabs(alpha[i]) > 0)
			++nSV;
	}

	info("Objective value = %lf\n",v);
	info("nSV = %d\n",nSV);

	delete [] alpha;
	delete [] h;
	delete [] index;
	delete [] QD;
	delete [] y_index;

}

*/



//shrinking
static void solve_SSVOR(const problem *prob, int nr_class,  double *w, double eps, double C)
{
	int tl = prob->l/5;
	int l = prob->l - tl;
	// int ncount = 0;
	// int ntime = prob->l/10;
	clock_t start, stop;
	start=clock();





		// info("%f %f\n",Cp,Cn);
	// int l = prob->l;
	int w_size = prob->n;
	int i,i0, s, m, iter = 0;
	double d, G;
	double *QD = new double[l];
	int max_iter = 1000;
	int *index = new int[l*(nr_class-1)];
	int *alpha_index = new int[l*(nr_class-1)];	
	double *alpha =  new double[l*(nr_class-1)];
	double *y = new double[l];
	int active_size = l*(nr_class-1);
	// PG: projected gradient, for shrinking and stopping
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration
	for(i=0; i<l; i++)
	{
		y[i] = prob->y[i];
		for(m=0; m<nr_class-1; m++)		
		{
			index[i*(nr_class-1)+m] = i;
			alpha_index[i*(nr_class-1)+m] = i*(nr_class-1)+m;
		}		
	}

	// Initial alpha can be set here. Note that
	// 0 <= alpha[i] <= upper_bound[GETI(i)]
	for(i=0; i<l*(nr_class-1); i++)
		alpha[i] = 0;

	for(i=0; i<w_size*nr_class; i++)
		w[i] = 0;
	for(i=0; i<l; i++)
	{
		QD[i] = 0;

		feature_node * const xi = prob->x[i];
		QD[i] += sparse_operator::nrm2_sq(xi);
		// sparse_operator::axpy(y[i]*alpha[i], xi, w);
	}
	active_size = l*(nr_class-1);	
	while (iter < max_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;

		for (i=0; i<active_size; i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
			swap(alpha_index[i], alpha_index[j]);
		}

		for (s=0; s<active_size; s++)
		{

			i = index[s];
			i0 = alpha_index[s];
			double yi = y[i];
			
			m = i0%(nr_class-1);
			double yim = (yi>m)?1:-1;
			double violation = 0;
			G = -1;
			// info("SSVOR1 m=%d i=%d ",m,index[s]);
			feature_node *xi = prob->x[i];
			if(QD[i]==0)
				continue;
			while(xi->index!= -1)
			{
				double *w_i = &w[(xi->index-1)*nr_class];
				G += yim*(w_i[m]+w_i[m+1])*(xi->value);
				xi++;
			}
			// info("SSVOR2 G= %.2f iter = %d index[s]=%d\n",G, iter,index[s]);			
			if (alpha[i0] == 0)
			{
				if (G > Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					swap(alpha_index[s], alpha_index[active_size]);
					s--;
					continue;
				}
				else if (G < 0)
					violation = -G;
			}
			else if (alpha[i0] == C)
			{
				if (G < -Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					swap(alpha_index[s], alpha_index[active_size]);
					s--;
					continue;
				}
				else if (G > 0)
					violation = G;
			}
			else
				violation = fabs(G);
			Gmax_new = max(Gmax_new, violation);
			Gnorm1_new += violation;
			if(fabs(G) > 1.0e-12)
			{
				double alpha_old = alpha[i0];
				alpha[i0] = min(max(alpha[i0] - G/QD[i], 0.0), C);
				d = (alpha[i0] - alpha_old)*yim;
				feature_node *xi = prob->x[i];
				while(xi->index != -1)
				{
					double *w_i = &w[(xi->index-1)*nr_class];
					w_i[m] += d*xi->value;
					w_i[m+1] += d*xi->value;
					xi++;
				}				

			}

			
		}
		if(iter == 0)
			Gnorm1_init = Gnorm1_new;
		iter++;
		if(iter % 10 == 0)
			info(".");

		if(Gnorm1_new <= eps*Gnorm1_init)
		{
			if(active_size == l*(nr_class-1))
				break;
			else
			{
				active_size = l*(nr_class-1);				
				info("*");
				Gmax_old = INF;
				continue;
			}
		}
		Gmax_old = Gmax_new;



		///measures vs time
		{
			int nr_w = nr_class;
			// int t;
			double mze=0;
			double mae = 0.0, mse = 0.0, T, diff,target_label,predict_y;	

			for(int t=l;t<prob->l;t++)		
				{
					target_label = (double)prob->y[t];
					predict_y = (double)predict_label(prob->x[t], w, prob->label, nr_w, w_size, nr_class);
					// info("target_label=%.3f predict_y=%.3f\n",target_label, predict_y);
					if(predict_y != target_label)
						mze += 1;
					diff = fabs(predict_y -target_label);
					mae += diff;
					mse += diff*diff;
					// printf("%d %.3f %.3f\n",mze, mae, diff);
				}				
			stop=clock();
			T = (double)(stop-start)/CLOCKS_PER_SEC;
			printf("%.4f %.3f %.3f %.3f\n",T, mze/tl, mae/tl, mse/tl);

		}


		// double v = 0;
		// int nSV = 0;
		// for(i=0; i<w_size*nr_class; i++)
		// 	v += w[i]*w[i];
		// v = v/2;
		// for(i=0; i<l*(nr_class-1); i++)
		// {
		// 	v -= alpha[i];
		// 	if(alpha[i] > 0)
		// 		++nSV;
		// }
		// info("Objective value = %lf nSV = %d\n",v, nSV);
	}

	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\nUsing -s 2 may be faster (also see FAQ)\n\n");

	// calculate objective value

	double v = 0;
	int nSV = 0;
	for(i=0; i<w_size*nr_class; i++)
		v += w[i]*w[i];
	v = v/2;
	for(i=0; i<l*(nr_class-1); i++)
	{
		v -= alpha[i];
		if(alpha[i] > 0)
			++nSV;
	}
	info("Objective value = %lf nSV = %d\n",v, nSV);
	delete [] QD;
	delete [] alpha;
	delete [] y;
	delete [] index;
	delete [] alpha_index;
}



// //shrinking
// static void solve_SSVOR(const problem *prob, int nr_class,  double *w, double eps, double C)
// {
// 		// info("%f %f\n",Cp,Cn);

// 	double tl = 0.2*prob->l;
// 	int l = prob->l - (int)tl;
// 	int ncount = 0;
// 	int ntime = prob->l/10;
// 	clock_t start, stop;
// 	start=clock();
// 	// double label = new double[nr_class];
//  //    for(int k=0; k<nr_class; nr_class)
//  //    	label[k] = prob->label[k];

// 	int w_size = prob->n;
// 	int i,i0, s, m, iter = 0;
// 	double d, G;
// 	double *QD = new double[l];
// 	int max_iter = 1000;
// 	int *index = new int[l*(nr_class-1)];
// 	int *alpha_index = new int[l*(nr_class-1)];	
// 	double *alpha =  new double[l*(nr_class-1)];
// 	double *y = new double[l];
// 	int active_size = l*(nr_class-1);

// 	// PG: projected gradient, for shrinking and stopping
// 	double Gmax_old = INF;
// 	double Gmax_new, Gnorm1_new;
// 	double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration

// 	for(i=0; i<l; i++)
// 	{
// 		y[i] = prob->y[i];
// 		for(m=0; m<nr_class-1; m++)		
// 		{
// 			index[i*(nr_class-1)+m] = i;
// 			alpha_index[i*(nr_class-1)+m] = i*(nr_class-1)+m;
// 		}		
// 	}


// 	// Initial alpha can be set here. Note that
// 	// 0 <= alpha[i] <= upper_bound[GETI(i)]
// 	for(i=0; i<l*(nr_class-1); i++)
// 		alpha[i] = 0;

// 	for(i=0; i<w_size*nr_class; i++)
// 		w[i] = 0;
// 	for(i=0; i<l; i++)
// 	{
// 		QD[i] = 0;

// 		feature_node * const xi = prob->x[i];
// 		QD[i] += sparse_operator::nrm2_sq(xi);
// 		// sparse_operator::axpy(y[i]*alpha[i], xi, w);
// 	}
// 	active_size = l*(nr_class-1);	




// 	while (iter < max_iter)
// 	{
		
// 		Gmax_new = 0;
// 		Gnorm1_new = 0;

// 		for (i=0; i<active_size; i++)
// 		{
// 			int j = i+rand()%(active_size-i);
// 			swap(index[i], index[j]);
// 			swap(alpha_index[i], alpha_index[j]);
// 		}

// 		for (s=0; s<active_size; s++)
// 		{

// 			i = index[s];
// 			i0 = alpha_index[s];
// 			double yi = y[i];
			
// 			m = i0%(nr_class-1);
// 			double yim = (yi>m)?1:-1;
// 			double violation = 0;
// 			G = -1;
// 			// info("SSVOR1 m=%d i=%d ",m,index[s]);
// 			feature_node *xi = prob->x[i];
// 			if(QD[i]==0)
// 				continue;
// 			while(xi->index!= -1)
// 			{
// 				double *w_i = &w[(xi->index-1)*nr_class];
// 				G += yim*(w_i[m]+w_i[m+1])*(xi->value);
// 				xi++;
// 			}
// 			// info("SSVOR2 G= %.2f iter = %d index[s]=%d\n",G, iter,index[s]);			
// 			if (alpha[i0] == 0)
// 			{
// 				if (G > Gmax_old)
// 				{
// 					active_size--;
// 					swap(index[s], index[active_size]);
// 					swap(alpha_index[s], alpha_index[active_size]);
// 					s--;
// 					continue;
// 				}
// 				else if (G < 0)
// 					violation = -G;
// 			}
// 			else if (alpha[i0] == C)
// 			{
// 				if (G < -Gmax_old)
// 				{
// 					active_size--;
// 					swap(index[s], index[active_size]);
// 					swap(alpha_index[s], alpha_index[active_size]);
// 					s--;
// 					continue;
// 				}
// 				else if (G > 0)
// 					violation = G;
// 			}
// 			else
// 				violation = fabs(G);
// 			Gmax_new = max(Gmax_new, violation);
// 			Gnorm1_new += violation;
// 			if(fabs(G) > 1.0e-12)
// 			{
// 				double alpha_old = alpha[i0];
// 				alpha[i0] = min(max(alpha[i0] - G/QD[i], 0.0), C);
// 				d = (alpha[i0] - alpha_old)*yim;
// 				feature_node *xi = prob->x[i];
// 				while(xi->index != -1)
// 				{
// 					double *w_i = &w[(xi->index-1)*nr_class];
// 					w_i[m] += d*xi->value;
// 					w_i[m+1] += d*xi->value;
// 					xi++;
// 				}				

// 			}

	

// 		}
// 		if(iter == 0)
// 			Gnorm1_init = Gnorm1_new;
// 		iter++;
// 		if(iter % 10 == 0)
// 			info(".");

// 		if(Gnorm1_new <= eps*Gnorm1_init)
// 		{
// 			if(active_size == l*(nr_class-1))
// 				break;
// 			else
// 			{
// 				active_size = l*(nr_class-1);
// 				info("*");
// 				Gmax_old = INF;
// 				continue;
// 			}
// 		}
// 		Gmax_old = Gmax_new;


// 		// double v = 0;
// 		// int nSV = 0;
// 		// for(i=0; i<w_size*nr_class; i++)
// 		// 	v += w[i]*w[i];
// 		// v = v/2;
// 		// for(i=0; i<l*(nr_class-1); i++)
// 		// {
// 		// 	v -= alpha[i];
// 		// 	if(alpha[i] > 0)
// 		// 		++nSV;
// 		// }
// 		// info("Objective value = %lf nSV = %d\n",v, nSV);


// 		///measures vs time
// 		{
// 			int nr_w = nr_class;
// 			int t;
// 			int mze=0;
// 			double mae = 0.0, mse = 0.0, T, diff;	

// 			for(int t=l;t<prob->l;t++)		
// 				{
// 					double target_label = prob->y[t];
// 					double 	predict_y = predict_label(prob->x[t], w, prob->label, nr_w, w_size, nr_class);
// 					// info("target_label=%.3f predict_y=%.3f\n",target_label, predict_y);
// 					if(predict_y != target_label)
// 						mze += 1;
// 					diff = fabs(predict_y -target_label);
// 					mae += diff;
// 					mse += diff*diff;
// 					printf("%.3f %.3f %.3f\n",mze, mae, diff);
// 				}				
// 			stop=clock();
// 			T = (double)(stop-start)/CLOCKS_PER_SEC;
// 			// printf("%.4f %.3f %.3f %.3f\n",T, mze, mae, mse);

// 		}
      

// 	}

// 	info("\noptimization finished, #iter = %d\n",iter);
// 	if (iter >= max_iter)
// 		info("\nWARNING: reaching max number of iterations\nUsing -s 2 may be faster (also see FAQ)\n\n");

// 	// calculate objective value

// 	double v = 0;
// 	int nSV = 0;
// 	for(i=0; i<w_size*nr_class; i++)
// 		v += w[i]*w[i];
// 	v = v/2;
// 	for(i=0; i<l*(nr_class-1); i++)
// 	{
// 		v -= alpha[i];
// 		if(alpha[i] > 0)
// 			++nSV;
// 	}
// 	info("Objective value = %lf nSV = %d\n",v, nSV);
// 	delete [] QD;
// 	delete [] alpha;
// 	delete [] y;
// 	delete [] index;
// 	delete [] alpha_index;
// }











static void solve_SNPSVOR(const problem *prob, int nr_class,  double *w, const parameter *param)
{
		// info("%f %f\n",Cp,Cn);
	int l = prob->l;
	int w_size = prob->n;
	int i,i0, s, m, iter = 0;
	double d, G;
	double *QD = new double[l];
	int max_iter = 1000;
	int *index = new int[l*nr_class];
	int *alpha_index = new int[l*nr_class];	
	double *alpha =  new double[l*nr_class];
	double *y = new double[l];
	int active_size = l*nr_class;

	// PG: projected gradient, for shrinking and stopping
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration


	double C1 = param->C1;
	double C2 = param->C2;
	double p = param->p;
	double eps = param->eps;

	for(i=0; i<l; i++)
	{
		y[i] = prob->y[i];
		for(m=0; m<nr_class; m++)		
		{
			index[i*nr_class+m] = i;
			alpha_index[i*nr_class+m] = i*nr_class+m;
		}		
	}

	// Initial alpha can be set here. Note that
	// 0 <= alpha[i] <= upper_bound[GETI(i)]
	for(i=0; i<l*nr_class; i++)
		alpha[i] = 0;

	for(i=0; i<w_size*nr_class; i++)
		w[i] = 0;
	for(i=0; i<l; i++)
	{
		QD[i] = 0;

		feature_node * const xi = prob->x[i];
		QD[i] += sparse_operator::nrm2_sq(xi);
		// sparse_operator::axpy(y[i]*alpha[i], xi, w);
	}
	active_size = l*nr_class;	
	while (iter < max_iter)
	{

		Gmax_new = 0;
		Gnorm1_new = 0;

		for (i=0; i<active_size; i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
			swap(alpha_index[i], alpha_index[j]);
		}

		for (s=0; s<active_size; s++)
		{
			i = index[s];
			i0 = alpha_index[s];
			int yi = (int)y[i];
			feature_node *xi = prob->x[i];
			if(QD[i]==0)
				continue;			
			m = i0%nr_class;
			double yim = (yi>m)?1:-1;
			double violation = 0;

			if(m==nr_class-1)
			{

				G = 0;	
				while(xi->index!= -1)
				{
					double *w_i = &w[(xi->index-1)*nr_class];
					G += yim*w_i[yi]*(xi->value);
					xi++;
				}

				double Gp = G+p;
				double Gn = G-p;

				if(alpha[i0] == 0)
				{
					if(Gp < 0)
						violation = -Gp;
					else if(Gn > 0)
						violation = Gn;
					else if(Gp>Gmax_old && Gn<-Gmax_old)
					{
						active_size--;
						swap(index[s], index[active_size]);
						swap(alpha_index[s], alpha_index[active_size]);
						s--;
						continue;
					}
				}
				else if(alpha[i0] >= C1)
				{
					if(Gp > 0)
						violation = Gp;
					else if(Gp < -Gmax_old)
					{
						active_size--;
						swap(index[s], index[active_size]);
						swap(alpha_index[s], alpha_index[active_size]);
						s--;
						continue;
					}
				}
				else if(alpha[i0] <= -C1)
				{
					if(Gn < 0)
						violation = -Gn;
					else if(Gn > Gmax_old)
					{
						active_size--;
						swap(index[s], index[active_size]);
						swap(alpha_index[s], alpha_index[active_size]);
						s--;
						continue;
					}
				}
				else if(alpha[i0] > 0)
					violation = fabs(Gp);
				else
					violation = fabs(Gn);

				Gmax_new = max(Gmax_new, violation);
				Gnorm1_new += violation;

				// obtain Newton direction d
				if(Gp < QD[i]*alpha[i0])
					d = -Gp/QD[i];
				else if(Gn > QD[i]*alpha[i0])
					d = -Gn/QD[i];
				else
					d = -alpha[i0];

				if(fabs(violation) > 1.0e-12)
				{
					double alpha_old = alpha[i0];
					alpha[i0] = min(max(alpha[i0]+d, -C1), C1);
					d = yim*(alpha[i0]-alpha_old);
					feature_node *xi = prob->x[i];
					while(xi->index != -1)
					{
						double *w_i = &w[(xi->index-1)*nr_class];
						w_i[yi] += d*xi->value;
						xi++;
					}
				}
			}
			else
			{
				G = -1;
				// info("SSVOR1 m=%d i=%d ",m,index[s]);
				while(xi->index!= -1)
				{
					double *w_i = &w[(xi->index-1)*nr_class];
					G += yim*(w_i[m]+w_i[m+1])*(xi->value);
					xi++;
				}
				// info("SSVOR2 G= %.2f iter = %d index[s]=%d\n",G, iter,index[s]);			
				if (alpha[i0] == 0)
				{
					if (G > Gmax_old)
					{
						active_size--;
						swap(index[s], index[active_size]);
						swap(alpha_index[s], alpha_index[active_size]);
						s--;
						continue;
					}
					else if (G < 0)
						violation = -G;
				}
				else if (alpha[i0] == C2)
				{
					if (G < -Gmax_old)
					{
						active_size--;
						swap(index[s], index[active_size]);
						swap(alpha_index[s], alpha_index[active_size]);
						s--;
						continue;
					}
					else if (G > 0)
						violation = G;
				}
				else
					violation = fabs(G);
				Gmax_new = max(Gmax_new, violation);
				Gnorm1_new += violation;
				if(fabs(G) > 1.0e-12)
				{
					double alpha_old = alpha[i0];
					alpha[i0] = min(max(alpha[i0] - G/QD[i], 0.0), C2);
					d = (alpha[i0] - alpha_old)*yim;
					feature_node *xi = prob->x[i];
					while(xi->index != -1)
					{
						double *w_i = &w[(xi->index-1)*nr_class];
						w_i[m] += d*xi->value;
						w_i[m+1] += d*xi->value;
						xi++;
					}				

				}				
			}


			
		}
		if(iter == 0)
			Gnorm1_init = Gnorm1_new;
		iter++;
		if(iter % 10 == 0)
			info(".");



		// double v = 0;
		// int nSV = 0;
		// for(i=0; i<w_size*nr_class; i++)
		// 	v += w[i]*w[i];
		// v = v/2;
		// for(i=0; i<l*nr_class; i++)
		// {
		// 	if(i%nr_class == nr_class-1)
		// 	{
		// 		if(fabs(alpha[i]) > 0)
		// 			++nSV;	
		// 	}
		// 	else
		// 	{
		// 		v -= alpha[i];
		// 		if(alpha[i] > 0)
		// 			++nSV;				
		// 	}
		// }
		// info("Objective value = %lf nSV = %d\n",v, nSV);





		if(Gnorm1_new <= eps*Gnorm1_init)
		{
			if(active_size == l*nr_class)
				break;
			else
			{
				active_size = l*nr_class;
				info("*");
				Gmax_old = INF;
				continue;
			}
		}
		Gmax_old = Gmax_new;

	}

	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\nUsing -s 2 may be faster (also see FAQ)\n\n");

	// calculate objective value

	double v = 0;
	int nSV = 0;
	for(i=0; i<w_size*nr_class; i++)
		v += w[i]*w[i];
	v = v/2;
	for(i=0; i<l*nr_class; i++)
	{
		if(i%nr_class == nr_class-1)
		{
			if(fabs(alpha[i]) > 0)
				++nSV;	
		}
		else
		{
			v -= alpha[i];
			if(alpha[i] > 0)
				++nSV;				
		}
	}
	info("Objective value = %lf nSV = %d\n",v, nSV);
	delete [] QD;
	delete [] alpha;
	delete [] y;
	delete [] index;
	delete [] alpha_index;
}






// A coordinate descent algorithm for 
// L1-loss and L2-loss SVM dual problems
//
//  min_\alpha  0.5(\alpha^T (Q + D)\alpha) - e^T \alpha,
//    s.t.      0 <= \alpha_i <= upper_bound_i,
// 
//  where Qij = yi yj xi^T xj and
//  D is a diagonal matrix 
//
// In L1-SVM case:
// 		upper_bound_i = Cp if y_i = 1
// 		upper_bound_i = Cn if y_i = -1
// 		D_ii = 0
// In L2-SVM case:
// 		upper_bound_i = INF
// 		D_ii = 1/(2*Cp)	if y_i = 1
// 		D_ii = 1/(2*Cn)	if y_i = -1
//
// Given: 
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
// 
// See Algorithm 3 of Hsieh et al., ICML 2008

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

static void solve_l2r_l1l2_svc(
	const problem *prob, double *w, double eps,
	double Cp, double Cn, int solver_type)
{
		// info("%f %f\n",Cp,Cn);
	int l = prob->l;
	int w_size = prob->n;
	int i, s, iter = 0;
	double C, d, G;
	double *QD = new double[l];
	int max_iter = 1000;
	int *index = new int[l];
	double *alpha = new double[l];
	schar *y = new schar[l];
	int active_size = l;

	// PG: projected gradient, for shrinking and stopping
	double PG;
	double PGmax_old = INF;
	double PGmin_old = -INF;
	double PGmax_new, PGmin_new;

	// default solver_type: L2R_L2LOSS_SVC_DUAL
	double diag[3] = {0.5/Cn, 0, 0.5/Cp};
	double upper_bound[3] = {INF, 0, INF};
	if(solver_type == L2R_L1LOSS_SVC_DUAL)
	{
		diag[0] = 0;
		diag[2] = 0;
		upper_bound[0] = Cn;
		upper_bound[2] = Cp;
	}

	for(i=0; i<l; i++)
	{
		if(prob->y[i] > 0)
		{
			y[i] = +1;
		}
		else
		{
			y[i] = -1;
		}
	}

	// Initial alpha can be set here. Note that
	// 0 <= alpha[i] <= upper_bound[GETI(i)]
	for(i=0; i<l; i++)
		alpha[i] = 0;

	for(i=0; i<w_size; i++)
		w[i] = 0;
	for(i=0; i<l; i++)
	{
		QD[i] = diag[GETI(i)];

		feature_node * const xi = prob->x[i];
		QD[i] += sparse_operator::nrm2_sq(xi);
		sparse_operator::axpy(y[i]*alpha[i], xi, w);

		index[i] = i;
	}

	while (iter < max_iter)
	{
		PGmax_new = -INF;
		PGmin_new = INF;

		for (i=0; i<active_size; i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}

		for (s=0; s<active_size; s++)
		{
			i = index[s];
			const schar yi = y[i];
			feature_node * const xi = prob->x[i];

			G = yi*sparse_operator::dot(w, xi)-1;

			C = upper_bound[GETI(i)];
			G += alpha[i]*diag[GETI(i)];

			PG = 0;
			if (alpha[i] == 0)
			{
				if (G > PGmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G < 0)
					PG = G;
			}
			else if (alpha[i] == C)
			{
				if (G < PGmin_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G > 0)
					PG = G;
			}
			else
				PG = G;

			PGmax_new = max(PGmax_new, PG);
			PGmin_new = min(PGmin_new, PG);

			if(fabs(PG) > 1.0e-12)
			{
				double alpha_old = alpha[i];
				alpha[i] = min(max(alpha[i] - G/QD[i], 0.0), C);
				d = (alpha[i] - alpha_old)*yi;
				sparse_operator::axpy(d, xi, w);
			}
		}

		iter++;
		if(iter % 10 == 0)
			info(".");

		if(PGmax_new - PGmin_new <= eps)
		{
			if(active_size == l)
				break;
			else
			{
				active_size = l;
				info("*");
				PGmax_old = INF;
				PGmin_old = -INF;
				continue;
			}
		}
		PGmax_old = PGmax_new;
		PGmin_old = PGmin_new;
		if (PGmax_old <= 0)
			PGmax_old = INF;
		if (PGmin_old >= 0)
			PGmin_old = -INF;
	}

	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\nUsing -s 2 may be faster (also see FAQ)\n\n");

	// calculate objective value

	double v = 0;
	int nSV = 0;
	for(i=0; i<w_size; i++)
		v += w[i]*w[i];
	for(i=0; i<l; i++)
	{
		v += alpha[i]*(alpha[i]*diag[GETI(i)] - 2);
		if(alpha[i] > 0)
			++nSV;
	}
	info("Objective value = %lf\n",v/2);
	//info("nSV = %d\n",nSV);
	//info("Percentage of SVs:%f \n",(double)nSV*100/l);
	info(" %f  \n",(double)nSV*100/l);
	delete [] QD;
	delete [] alpha;
	delete [] y;
	delete [] index;
}


static void solve_l2r_l1l2_svmop(
	const problem *prob, double *w, double eps,
	double Cp, double Cn, int solver_type)
{
		// info("%f %f\n",Cp,Cn);
	int l = prob->l;
	int w_size = prob->n;
	int i, s, iter = 0;
	double C, d, G;
	double *QD = new double[l];
	int max_iter = 1000;
	int *index = new int[l];
	double *alpha = new double[l];
	schar *y = new schar[l];
	int active_size = l;

	// PG: projected gradient, for shrinking and stopping
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration
	// default solver_type: L2R_L2LOSS_SVC_DUAL
	double diag[3] = {0.5/Cn, 0, 0.5/Cp};
	double upper_bound[3] = {INF, 0, INF};
	if(solver_type == L2R_L1LOSS_SVC_DUAL)
	{
		diag[0] = 0;
		diag[2] = 0;
		upper_bound[0] = Cn;
		upper_bound[2] = Cp;
	}

	for(i=0; i<l; i++)
	{
		if(prob->y[i] > 0)
		{
			y[i] = +1;
		}
		else
		{
			y[i] = -1;
		}
	}

	// Initial alpha can be set here. Note that
	// 0 <= alpha[i] <= upper_bound[GETI(i)]
	for(i=0; i<l; i++)
		alpha[i] = 0;

	for(i=0; i<w_size; i++)
		w[i] = 0;
	for(i=0; i<l; i++)
	{
		QD[i] = diag[GETI(i)];

		feature_node * const xi = prob->x[i];
		QD[i] += sparse_operator::nrm2_sq(xi);
		// sparse_operator::axpy(y[i]*alpha[i], xi, w);

		index[i] = i;
	}

	while (iter < max_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;

		for (i=0; i<active_size; i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}

		for (s=0; s<active_size; s++)
		{
			i = index[s];
			const schar yi = y[i];
			feature_node * const xi = prob->x[i];
			double violation = 0;

			G = yi*sparse_operator::dot(w, xi)-1;

			C = upper_bound[GETI(i)];
			G += alpha[i]*diag[GETI(i)];

			if (alpha[i] == 0)
			{
				if (G > Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G < 0)
					violation = -G;
			}
			else if (alpha[i] == C)
			{
				if (G < -Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G > 0)
					violation = G;
			}
			else
				violation = fabs(G);

			Gmax_new = max(Gmax_new, violation);
			Gnorm1_new += violation;

			if(fabs(G) > 1.0e-12)
			{
				double alpha_old = alpha[i];
				alpha[i] = min(max(alpha[i] - G/QD[i], 0.0), C);
				d = (alpha[i] - alpha_old)*yi;
				sparse_operator::axpy(d, xi, w);
			}
		}
		if(iter == 0)
			Gnorm1_init = Gnorm1_new;
		iter++;
		if(iter % 10 == 0)
			info(".");

		if(Gnorm1_new <= eps*Gnorm1_init)
		{
			if(active_size == l)
				break;
			else
			{
				active_size = l;
				info("*");
				Gmax_old = INF;
				continue;
			}
		}
		Gmax_old = Gmax_new;
	}

	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\nUsing -s 2 may be faster (also see FAQ)\n\n");

	// calculate objective value

	double v = 0;
	int nSV = 0;
	for(i=0; i<w_size; i++)
		v += w[i]*w[i];
	for(i=0; i<l; i++)
	{
		v += alpha[i]*(alpha[i]*diag[GETI(i)] - 2);
		if(alpha[i] > 0)
			++nSV;
	}
	info("Objective value = %lf\n",v/2);
	//info("nSV = %d\n",nSV);
	//info("Percentage of SVs:%f \n",(double)nSV*100/l);
	info(" %f  \n",(double)nSV*100/l);
	delete [] QD;
	delete [] alpha;
	delete [] y;
	delete [] index;
}




/*
* ------------------modification begin---------------------------
*/
int calculate_y_k(double y, int k)
{
	if(y>k+1){
		return 1;
	}else{
		return -1;
	}
}
double compute_nu_i_k(double y, int k, double power){
	if(power==0) return 1;
		else return fabs(pow(fabs(double(y- k)), power) - pow(fabs(double(y- k-1)), power));
}

static void solve_l2r_svor(const problem *prob, const parameter *param, double *w,
	double *b, int *label, int nr_class)//cost refers to power
{
	int i, j, k, s, iter = 0;
	int l = prob->l;
	// int nr_class = 0;//number of classes
	// int max_nr_class = 16;//max number of classes
	// int *label = new int[max_nr_class];//category of labels
	double *y = prob->y;
	// int this_label = 0;
	// for(i=0;i<l;i++)
	// {
	// 	this_label = (int)prob->y[i];
	// 	for(j=0;j<nr_class;j++)
	// 	{
	// 		if(this_label == label[j])
	// 			break;
	// 	}
	// 	y[i] = this_label;
	// 	if(j == nr_class)
	// 	{
	// 		label[nr_class] = this_label;
	// 		++nr_class;
	// 	}
	// }
	double eps = param->eps;
	double C = param->C;
	double power = param->wl;
	// int w_size = prob->n;
	double d, G;
	double *QD = new double[l];
	int max_iter = 1000;
	int *index = new int[(nr_class - 1)*l];
	double *alpha = new double[(nr_class - 1)*l];
	int active_size = (nr_class - 1)*l;
	// PG: projected gradient, for shrinking and stopping
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration


/*
	//to be check
	// default solver_type: L2R_L2LOSS_SVC_DUAL
	double diag[3] = {0.5/Cn, 0, 0.5/Cp};
	double upper_bound[3] = {INF, 0, INF};
	if(solver_type == L2R_L1LOSS_SVC_DUAL)
	{
		diag[0] = 0;
		diag[2] = 0;
		upper_bound[0] = Cn;
		upper_bound[2] = Cp;
	}
*/


	// Initial alpha can be set here. Note that
	// 0 <= alpha[i] <= upper_bound[GETI(i)]


	memset(alpha,0,sizeof(double)*((nr_class - 1)*l));

	for(i=0; i<l; i++)	
	{
		feature_node * const xi = prob->x[i];
		double xi_square = sparse_operator::nrm2_sq(xi);
		QD[i] = xi_square+1; // add
		for(k= 0; k<nr_class-1; k++)
		{
			//int y_i_k = calculate_y_k(y[i], k+1);		
			//QD[k*l + i] = y_i_k*y_i_k*(xi_square+1);
			// QD[k*l + i] = xi_square+1;
//			sparse_operator::axpy(y_i_k*alpha[i*nr_class + k], xi, w);
			index[k*l + i] = k*l + i;
		}
	}
	
	int kk, ss;//kk(k'),ss(s) in the paper
	// int i0,kk0,ss0,t;
	while (iter < max_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;

		for (i=0; i<active_size; i++)
		{
			j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}

		for (s=0; s<active_size; s++)
		{
			i = index[s];
			kk = i/l;//k'
			ss = i%l;//s
			int y_s_ksign = (prob->y[ss] <= label[kk] ? -1 : 1);//ysk'
			feature_node * const xss = prob->x[ss];

			G = y_s_ksign*(sparse_operator::dot(w, xss)+b[kk])-1;

			//C = upper_bound[GETI(i)];
			//G += alpha[i]*diag[GETI(i)];
			double violation = 0;
			double upper_bound = C*compute_nu_i_k(y[ss], label[kk], power);
			if (alpha[i] == 0)
			{
				if (G > Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G < 0)
					violation = -G;
			}
			else if (alpha[i] == upper_bound)
			{
				if (G < -Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;				
					continue;
				}
				else if (G > 0)
					violation = G;
			}
			else
				violation = fabs(G);


			Gmax_new = max(Gmax_new, violation);
			Gnorm1_new += violation;

			if(fabs(G) > 1.0e-12)
			{
				double alpha_old = alpha[i];
				alpha[i] = min(max(alpha[i] - G/QD[ss], 0.0), upper_bound);
				d = (alpha[i] - alpha_old)*y_s_ksign;
				sparse_operator::axpy(d, xss, w);
				b[kk] += d;
			}
		}
		// printf("%d ",active_size);
		if(iter == 0)
			Gnorm1_init = Gnorm1_new;
		iter++;
		if(iter % 10 == 0)
			info(".");

		if(Gnorm1_new <= eps*Gnorm1_init)
		{
			if(active_size == (nr_class - 1)*l)
				break;
			else
			{
				active_size = (nr_class - 1)*l;
				info("*");
				Gmax_old = INF;
				continue;
			}
		}
		Gmax_old = Gmax_new;
	}

	info("\noptimization finished, #iter = %d\n",iter);
	// for(i=0;i<nr_class-1;i++) info("%.6f\n",b[i]);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\nUsing -s 2 may be faster (also see FAQ)\n\n");

	// calculate objective value
	//double v = 0;
/*
	for(i=0; i<w_size; i++)
		v += w[i]*w[i];
	for(i=0; i<(nr_class - 1)*l; i++)
	{
	v += alpha[i]*(alpha[i]*diag[GETI(i)] - 2);
		if(alpha[i] > 0)
			++nSV;
	}
	info("Objective value = %lf\n",v/2);
*/
	int nSV = 0;
	for(i=0; i<l; i++)	
	{
		double alpha_i = 0;
		for(k= 0; k<nr_class-1; k++)
		{
			alpha_i  += alpha[k*l + i];
		}
		if(alpha_i > 0)
			++nSV;
	}
	info("nSV = %d\n",nSV);

	delete [] QD;
	delete [] alpha;
	delete [] y;
	delete [] index;		
}


// static void solve_l2r_svor(const problem *prob, const parameter *param, double *w,
// 	double *b, int nr_class)//cost refers to power
// {
// 	int i, j, k, s, iter = 0;
// 	int l = prob->l;
// 	// int nr_class = 0;//number of classes
// 	// int max_nr_class = 16;//max number of classes
// 	// int *label = new int[max_nr_class];//category of labels
// 	double *y = prob->y;
// 	// int this_label = 0;
// 	// for(i=0;i<l;i++)
// 	// {
// 	// 	this_label = (int)prob->y[i];
// 	// 	for(j=0;j<nr_class;j++)
// 	// 	{
// 	// 		if(this_label == label[j])
// 	// 			break;
// 	// 	}
// 	// 	y[i] = this_label;
// 	// 	if(j == nr_class)
// 	// 	{
// 	// 		label[nr_class] = this_label;
// 	// 		++nr_class;
// 	// 	}
// 	// }
// 	double eps = param->eps;
// 	double C = param->C;
// 	double power = param->wl;
// 	// int w_size = prob->n;
// 	double d, G;
// 	double *QD = new double[l];
// 	int max_iter = 1000;
// 	int *index = new int[(nr_class - 1)*l];
// 	double *alpha = new double[(nr_class - 1)*l];
// 	int active_size = (nr_class - 1)*l;
// 	// PG: projected gradient, for shrinking and stopping
// 	double PG;
// 	double PGmax_old = INF;
// 	double PGmin_old = -INF;
// 	double PGmax_new, PGmin_new;


// /*
// 	//to be check
// 	// default solver_type: L2R_L2LOSS_SVC_DUAL
// 	double diag[3] = {0.5/Cn, 0, 0.5/Cp};
// 	double upper_bound[3] = {INF, 0, INF};
// 	if(solver_type == L2R_L1LOSS_SVC_DUAL)
// 	{
// 		diag[0] = 0;
// 		diag[2] = 0;
// 		upper_bound[0] = Cn;
// 		upper_bound[2] = Cp;
// 	}
// */


// 	// Initial alpha can be set here. Note that
// 	// 0 <= alpha[i] <= upper_bound[GETI(i)]


// 	memset(alpha,0,sizeof(double)*((nr_class - 1)*l));
// 	// for(i=0; i<(nr_class - 1)*l; i++)
// 	// 	alpha[i] = 0;

// 	// for(i=0; i<w_size; i++)

// 	// 	w[i] = 0;
// 	// for(i=0; i<nr_class-1; i++)
// 	// 	b[i] = 0;
// 	for(i=0; i<l; i++)	
// 	{
// 		feature_node * const xi = prob->x[i];
// 		double xi_square = sparse_operator::nrm2_sq(xi);
// 		QD[i] = xi_square+1; // add
// 		for(k= 0; k<nr_class-1; k++)
// 		{
// 			//int y_i_k = calculate_y_k(y[i], k+1);		
// 			//QD[k*l + i] = y_i_k*y_i_k*(xi_square+1);
// 			// QD[k*l + i] = xi_square+1;
// //			sparse_operator::axpy(y_i_k*alpha[i*nr_class + k], xi, w);
// 			index[k*l + i] = k*l + i;
// 		}
// 	}
	
// 	int kk, ss;//kk(k'),ss(s) in the paper
// 	// int i0,kk0,ss0,t;
// 	while (iter < max_iter)
// 	{
// 		PGmax_new = -INF;
// 		PGmin_new = INF;

// 		for (i=0; i<active_size; i++)
// 		{
// 			j = i+rand()%(active_size-i);
// 			swap(index[i], index[j]);
// 		}

// 		for (s=0; s<active_size; s++)
// 		{
// 			i = index[s];
// 			kk = i/l;//k'
// 			ss = i%l;//s
// 			int y_s_ksign = calculate_y_k(y[ss], kk);//ysk'
// 			feature_node * const xss = prob->x[ss];

// 			G = y_s_ksign*(sparse_operator::dot(w, xss)+b[kk])-1;

// 			//C = upper_bound[GETI(i)];
// 			//G += alpha[i]*diag[GETI(i)];

// 			PG = 0;
// 			double upper_bound = C*compute_nu_i_k(y[ss], kk, power);
// 			if (alpha[i] == 0)
// 			{
// 				if (G > PGmax_old)
// 				{
// 					active_size--;
// 					swap(index[s], index[active_size]);
// 					s--;
// 					// add begin
// 					// if(kk<nr_class-1)
// 					// 	for(t=s;t<active_size;t++)
// 					// 	{
// 					// 		i0 = index[t];
// 					// 		kk0 = i0/l;//k'
// 					// 		ss0 = i0%l;//s
// 					// 		if(ss==ss0 && kk0>kk)
// 					// 			{
// 					// 				active_size--;
// 					// 				swap(index[t], index[active_size]);
// 					// 				n0++;
// 					// 			}
// 					// 		if(n0==(nr_class-1-kk0))break;
// 					// 	}
// 					//end
// 					continue;
// 				}
// 				else if (G < 0)
// 					PG = G;
// 			}
// 			else if (alpha[i] == upper_bound)
// 			{
// 				if (G < PGmin_old)
// 				{
// 					active_size--;
// 					swap(index[s], index[active_size]);
// 					s--;
// 					// add begin
// 					// if(kk>0)
// 					// {   n0=0;
// 					// 	for(t=s;t<active_size;t++)
// 					// 	{
// 					// 		i0 = index[t];
// 					// 		kk0 = i0/l;//k'
// 					// 		ss0 = i0%l;//s
// 					// 		if(ss==ss0 && kk0<kk)
// 					// 			{
// 					// 				active_size--;
// 					// 				swap(index[t], index[active_size]);
// 					// 				n0++;
// 					// 			}
// 					// 		if(n0==(nr_class-1-kk0))break;

// 					// 	}
// 					// }
// 					//end					
// 					continue;
// 				}
// 				else if (G > 0)
// 					PG = G;
// 			}
// 			else
// 				PG = G;


// 			PGmax_new = max(PGmax_new, PG);
// 			PGmin_new = min(PGmin_new, PG);

// 			if(fabs(PG) > 1.0e-12)
// 			{
// 				double alpha_old = alpha[i];
// 				alpha[i] = min(max(alpha[i] - G/QD[ss], 0.0), upper_bound);
// 				d = (alpha[i] - alpha_old)*y_s_ksign;
// 				sparse_operator::axpy(d, xss, w);
// 				b[kk] += d;
// 			}
// 		}
// 		// printf("%d ",active_size);

// 		iter++;
// 		if(iter % 10 == 0)
// 			info(".");

// 		if(PGmax_new - PGmin_new <= eps)
// 		{
// 			if(active_size == (nr_class - 1)*l)
// 				break;
// 			else
// 			{
// 				active_size = (nr_class - 1)*l;
// 				info("*");
// 				PGmax_old = INF;
// 				PGmin_old = -INF;
// 				continue;
// 			}
// 		}
// 		PGmax_old = PGmax_new;
// 		PGmin_old = PGmin_new;
// 		if (PGmax_old <= 0)
// 			PGmax_old = INF;
// 		if (PGmin_old >= 0)
// 			PGmin_old = -INF;
// 	}

// 	info("\noptimization finished, #iter = %d\n",iter);
// 	// for(i=0;i<nr_class-1;i++) info("%.6f\n",b[i]);
// 	if (iter >= max_iter)
// 		info("\nWARNING: reaching max number of iterations\nUsing -s 2 may be faster (also see FAQ)\n\n");

// 	// calculate objective value
// 	//double v = 0;
// /*
// 	for(i=0; i<w_size; i++)
// 		v += w[i]*w[i];
// 	for(i=0; i<(nr_class - 1)*l; i++)
// 	{
// 	v += alpha[i]*(alpha[i]*diag[GETI(i)] - 2);
// 		if(alpha[i] > 0)
// 			++nSV;
// 	}
// 	info("Objective value = %lf\n",v/2);
// */
// 	int nSV = 0;
// 	for(i=0; i<l; i++)	
// 	{
// 		double alpha_i = 0;
// 		for(k= 0; k<nr_class-1; k++)
// 		{
// 			alpha_i  += alpha[k*l + i];
// 		}
// 		if(alpha_i > 0)
// 			++nSV;
// 	}
// 	info("nSV = %d\n",nSV);

// 	delete [] QD;
// 	delete [] alpha;
// 	delete [] y;
// 	delete [] index;
// 	// for(i=0;i<prob->n;i++)
// 	// info("%.6f\n",w[i]);
// 	// for(i=0;i<nr_class-1;i++)
// 	// info("b%.6f\n",b[i]);		
// }



static void solve_l2r_svor_full(const problem *prob, const parameter *param, double *w,
	double *b, int *label, int nr_class)//cost refers to power
{
	int i, j, k, s, iter = 0;
	int l = prob->l;
	int bigl = l * (nr_class-1);
	schar *y = new schar[bigl];

	double eps = param->eps;
	double C = param->C;
	// int w_size = prob->n;
	double d, G;
	double *QD = new double[bigl];
	int max_iter = 1000;
	int *index = new int[bigl];
	double *alpha = new double[bigl];
	int active_size = bigl;
	// PG: projected gradient, for shrinking and stopping
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration

	int idx = 0;
	for(i=0;i<l;i++)
	{
		feature_node * const xi = prob->x[i];
		double xi_square = sparse_operator::nrm2_sq(xi);
	  for(k=0;k<nr_class -1;k++)
	  {
	    alpha[idx] = 0;
	    QD[idx] = xi_square+1;
	    y[idx] = (prob->y[i] <= label[k] ? -1 : 1);
	    index[idx] = idx;
	    idx++;
	  }
	}

    
	// memset(alpha,0,sizeof(double)*(bigl));
	
	int kk, ss;//kk(k'),ss(s) in the paper
	// int i0,kk0,ss0,t;
	while (iter < max_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;

		for (i=0; i<active_size; i++)
		{
			j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}
		
		for (s=0; s<active_size; s++)
		{
			i = index[s];
			ss = i/(nr_class-1);//
			kk = i%(nr_class-1);//s
			feature_node * const xss = prob->x[ss];
			G = y[i]*(sparse_operator::dot(w, xss)+b[kk])-1;

			//C = upper_bound[GETI(i)];
			//G += alpha[i]*diag[GETI(i)];
			double violation = 0;
			double upper_bound = C;
			if (alpha[i] == 0)
			{
				if (G > Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G < 0)
					violation = -G;
			}
			else if (alpha[i] == upper_bound)
			{
				if (G < -Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;				
					continue;
				}
				else if (G > 0)
					violation = G;
			}
			else
				violation = fabs(G);


			Gmax_new = max(Gmax_new, violation);
			Gnorm1_new += violation;

			if(fabs(G) > 1.0e-12)
			{
				double alpha_old = alpha[i];
				alpha[i] = min(max(alpha[i] - G/QD[i], 0.0), upper_bound);
				d = y[i]*(alpha[i] - alpha_old);
				sparse_operator::axpy(d, xss, w);
				b[kk] += d;
			}
		}
		// printf("%d ",active_size);
		if(iter == 0)
			Gnorm1_init = Gnorm1_new;
		iter++;
		if(iter % 10 == 0)
			info(".");

		if(Gnorm1_new <= eps*Gnorm1_init)
		{
			if(active_size == bigl)
				break;
			else
			{
				active_size = bigl;
				info("*");
				Gmax_old = INF;
				continue;
			}
		}
		Gmax_old = Gmax_new;
	}

	info("\noptimization finished, #iter = %d\n",iter);
	// for(i=0;i<nr_class-1;i++) info("%.6f\n",b[i]);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\nUsing -s 2 may be faster (also see FAQ)\n\n");

	// calculate objective value
	//double v = 0;
/*
	for(i=0; i<w_size; i++)
		v += w[i]*w[i];
	for(i=0; i<(nr_class - 1)*l; i++)
	{
	v += alpha[i]*(alpha[i]*diag[GETI(i)] - 2);
		if(alpha[i] > 0)
			++nSV;
	}
	info("Objective value = %lf\n",v/2);
*/
	int nSV = 0;
	for(i=0; i<l; i++)	
	{
		double alpha_i = 0;
		for(k= 0; k<nr_class-1; k++)
		{
			alpha_i  += alpha[i*(nr_class-1) + k];
		}
		if(alpha_i > 0)
			++nSV;
	}
	info("nSV = %d\n",nSV);

	delete [] QD;
	delete [] alpha;
	delete [] y;
	delete [] index;		
}


double npsvor_full_obj_value(const problem *prob,double *w,double *alpha, double p, int k)
{
	// calculate objective value
	double v = 0;
	double *y = prob->y;
	int i, l = prob->l,w_size = prob->n;
	for(i=0; i<w_size; i++)
		v += w[i]*w[i];
	v = 0.5*v;
	int j=0;
	for(i=0; i<l; i++)
	{
		if(y[i]== k)
		{
			v += p*(alpha[i]+alpha[l+j]);
		}
		else
			v += - alpha[i];
	}
	return v;	
}



static void solve_l2r_npsvor_full(
	const problem *prob, double *w, const parameter *param,
	int k, int nk)
{
	int l = prob->l;
	double C1 = param->C1;
	double C2 = param->C2;
	double p = param->p;
	double eps = param->eps;	
	int w_size = prob->n;
	int i,i0, s, iter = 0;
	double C, d, G;
	double *QD = new double[l];
	int max_iter = 1000;
	int *index = new int[l+nk];
	int *yk = new int[l+nk];
	double *alpha = new double[l+nk];
	double *y = prob->y;
	int active_size = l+nk;
	int *index0 = new int[l+nk];
	// PG: projected gradient, for shrinking and stopping
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration
	
	double T;
	// char file_name[1024];
	function *fun_obj=NULL;
	fun_obj=new l2r_l2_npsvor_fun(prob, param, k);

	// sprintf(file_name,"data_C1%g_C2%g_e%g_k%d_full.log",C1,C2,p,k);
	// FILE * log = fopen(file_name,"at+");
	clock_t start, stop;
	start=clock();

	// C2 = nk/(l-nk)*C2;
    
	// Initial alpha can be set here. Note that
	// 0 <= alpha[i] <= upper_bound[GETI(i)]
	memset(alpha,0,sizeof(double)*(l+nk));
	memset(w,0,sizeof(double)*w_size);
    int j=0;
	for(i=0; i<l; i++)
	{

		feature_node * const xi = prob->x[i];
		QD[i] = sparse_operator::nrm2_sq(xi);
		index[i] = i;
		index0[i] = i;
		if(y[i]<k)
			yk[i]=-1;
		else if(y[i]==k) 
			{
				yk[i] = -1;
				yk[l+j] =1;
				index[l+j] = l+j;
				index0[l+j] = i;
				j++;
			}
		else yk[i] = 1;

	}

	while (iter < max_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;

		for (i=0; i<active_size; i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}
		for (s=0; s<active_size; s++)
		{
			i = index[s];
			i0 = index0[i];
			feature_node * const xi = prob->x[i0];
			G = yk[i]*sparse_operator::dot(w, xi);

			if(y[i0]!=k)
				{G += -1; C = C2;}
			else
				{G += p; C = C1;}
			double violation = 0;
			if (alpha[i] == 0)
			{
				if (G > Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G < 0)
					violation = -G;
			}
			else if (alpha[i] == C)
			{
				if (G < -Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G > 0)
					violation = G;
			}
			else
				violation = fabs(G);

			Gmax_new = max(Gmax_new, violation);
			Gnorm1_new += violation;

			if(fabs(G) > 1.0e-12)
			{
				double alpha_old = alpha[i];
				alpha[i] = min(max(alpha[i] - G/QD[i0], 0.0), C);
				d = (alpha[i] - alpha_old)*yk[i];
				sparse_operator::axpy(d, xi, w);
			}
		}
		// printf("%d\n",nk);
		if(iter == 0)
			Gnorm1_init = Gnorm1_new;
		
		if(param->WithTime)
		{
			double v = 0;
			int nSV = 0;
			for(i=0; i<w_size; i++)
				v += w[i]*w[i];
			v = 0.5*v;
			for(i=0; i<l; i++)
			{
				if(y[i]== k)
					v += p*fabs(alpha[i]);
				else
					v += - alpha[i];
				if(alpha[i] != 0)
					nSV++;
			}
			double f;
			f = fun_obj->fun(w);
			// double funval = obj_value[iter];
			stop=clock();
			T = (double)(stop-start)/CLOCKS_PER_SEC;
			// fprintf(out,"%d\t%lf\t%g\n",iter,obj_value[iter],T[iter]);
			printf("%f %f %f %d\n",T,f,v,nSV);
		}

		iter++;
		// printf("%.3f ",max(PGmax_new,-PGmin_new));
		// if(iter % 10 == 0)
		// 	info(".");

		if(Gnorm1_new <= eps*Gnorm1_init)
		{
			if(active_size == l)
				break;
			else
			{
				active_size = l;
				// info("*");
				Gmax_old = INF;
				continue;
			}
		}
		Gmax_old = Gmax_new;
	}
	// fprintf(log, "\n\n");
	// if (NULL != log)
	//    fclose(log) ;	// clear the old file.	

	// info("\noptimization finished, #iter = %d\n",iter);
	// if (iter >= max_iter)
	// 	info("\nWARNING: reaching max number of iterations\nUsing -s 2 may be faster (also see FAQ)\n\n");

	// calculate objective value

	// calculate objective value
	double v = 0;
	int nSV = 0;
	for(i=0; i<w_size; i++)
		v += w[i]*w[i];
	v = 0.5*v;
	j=0;
	for(i=0; i<l; i++)
	{
		if(y[i]== k)
		{
			v += p*(alpha[i]+alpha[l+j]);
			if((alpha[i]-alpha[l+j]) != 0)
				nSV++;
			j++;
		}
		else
		{
			v += - alpha[i];
			if(alpha[i] != 0)
			nSV++;
		}
	}
	// info("Objective value = %lf\n",v/2);
	// info(" %f  \n",(double)nSV*100/l);
	delete [] QD;
	delete [] alpha;
	delete [] index;
	delete [] yk;
	delete [] index0;
}




static void solve_l2r_npsvor_full_Mm(
	const problem *prob, double *w, const parameter *param,
	int k, int nk)
{
	int l = prob->l;
	double C1 = param->C1;
	double C2 = param->C2;
	double p = param->p;
	double eps = param->eps;	
	int w_size = prob->n;
	int i,i0, s, iter = 0;
	double C, d, G;
	double *QD = new double[l];
	int max_iter = 1000;
	int *index = new int[l+nk];
	int *yk = new int[l+nk];
	double *alpha = new double[l+nk];
	double *y = prob->y;
	int active_size = l+nk;
	int *index0 = new int[l+nk];
	// PG: projected gradient, for shrinking and stopping
	double PG;
	double PGmax_old = INF;
	double PGmin_old = -INF;
	double PGmax_new, PGmin_new;

	double T;
	function *fun_obj=NULL;
	fun_obj=new l2r_l2_npsvor_fun(prob, param, k);

	clock_t start, stop;
	start=clock();
	// C2 = nk/(l-nk)*C2;
 

 // 	double *obj_value = new double[max_iter];
	// double *T = new double[max_iter];
	// char file_name[1024];
	// sprintf(file_name,"data_C1%g_C2%g_e%g_k%d_full_primal.log",C1,C2,p,k);
	// FILE * log = fopen(file_name,"at+");
	// clock_t start, stop;
	// start=clock();   
	
	// Initial alpha can be set here. Note that
	// 0 <= alpha[i] <= upper_bound[GETI(i)]
	memset(alpha,0,sizeof(double)*(l+nk));
	memset(w,0,sizeof(double)*w_size);
    int j=0;
	for(i=0; i<l; i++)
	{

		feature_node * const xi = prob->x[i];
		QD[i] = sparse_operator::nrm2_sq(xi);
		index[i] = i;
		index0[i] = i;
		if(y[i]<k)
			yk[i]=-1;
		else if(y[i]==k) 
			{
				yk[i] = -1;
				yk[l+j] =1;
				index[l+j] = l+j;
				index0[l+j] = i;
				j++;
			}
		else yk[i] = 1;

	}

	while (iter < max_iter)
	{
		PGmax_new = -INF;
		PGmin_new = INF;

		for (i=0; i<active_size; i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}
		for (s=0; s<active_size; s++)
		{
			i = index[s];
			i0 = index0[i];
			feature_node * const xi = prob->x[i0];
			G = yk[i]*sparse_operator::dot(w, xi);

			if(y[i0]!=k)
				{G += -1; C = C1;}
			else
				{G += p; C = C2;}
			PG = 0;
			if (alpha[i] == 0)
			{
				if (G > PGmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G < 0)
					PG = G;
			}
			else if (alpha[i] == C)
			{
				if (G < PGmin_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G > 0)
					PG = G;
			}
			else
				PG = G;

			PGmax_new = max(PGmax_new, PG);
			PGmin_new = min(PGmin_new, PG);

			if(fabs(PG) > 1.0e-12)
			{
				double alpha_old = alpha[i];
				alpha[i] = min(max(alpha[i] - G/QD[i0], 0.0), C);
				d = (alpha[i] - alpha_old)*yk[i];
				sparse_operator::axpy(d, xi, w);
			}
		}


		if(param->WithTime)
		{
			double v = 0;
			int nSV = 0;
			for(i=0; i<w_size; i++)
				v += w[i]*w[i];
			v = 0.5*v;
			for(i=0; i<l; i++)
			{
				if(y[i]== k)
					v += p*fabs(alpha[i]);
				else
					v += - alpha[i];
				if(alpha[i] != 0)
					nSV++;
			}
			double f;
			f = fun_obj->fun(w);
			// double funval = obj_value[iter];
			stop=clock();
			T = (double)(stop-start)/CLOCKS_PER_SEC;
			// fprintf(out,"%d\t%lf\t%g\n",iter,obj_value[iter],T[iter]);
			printf("%f %f %f %d\n",T,f,v,nSV);
		}



		iter++;


		// if(iter % 10 == 0)
		// 	info(".");

		if(PGmax_new - PGmin_new <= eps)
		{
			if(active_size == l)
				break;
			else
			{
				active_size = l;
				// info("*");
				PGmax_old = INF;
				PGmin_old = -INF;
				continue;
			}
		}
		PGmax_old = PGmax_new;
		PGmin_old = PGmin_new;
		if (PGmax_old <= 0)
			PGmax_old = INF;
		if (PGmin_old >= 0)
			PGmin_old = -INF;
	}

	// fprintf(log, "\n\n");
	// if (NULL != log)
	//    fclose(log) ;	// clear the old file.	

	// info("\noptimization finished, #iter = %d\n",iter);
	// if (iter >= max_iter)
	// 	info("\nWARNING: reaching max number of iterations\nUsing -s 2 may be faster (also see FAQ)\n\n");

	// calculate objective value

	// calculate objective value
	double v = 0;
	int nSV = 0;
	for(i=0; i<w_size; i++)
		v += w[i]*w[i];
	v = 0.5*v;
	j=0;
	for(i=0; i<l; i++)
	{
		if(y[i]== k)
		{
			v += p*(alpha[i]+alpha[l+j]);
			if((alpha[i]-alpha[l+j]) != 0)
				nSV++;
			j++;
		}
		else
		{
			v += - alpha[i];
			if(alpha[i] != 0)
			nSV++;
		}
	}
	// info("Objective value = %lf\n",v/2);
	// info(" %f  \n",(double)nSV*100/l);
	delete [] QD;
	delete [] alpha;
	delete [] index;
	delete [] yk;
	delete [] index0;
}


int calculate_yki(double y, int k)
{
	if(y>k){
		return 1;
	}else{
		return -1;
	}
}

double npsvor_obj_value(const problem *prob,double *w,double *alpha, double p, int k)
{
	// calculate objective value
	double v = 0;
	double *y = prob->y;
	int i, l = prob->l,w_size = prob->n;
	for(i=0; i<w_size; i++)
		v += w[i]*w[i];
	v = 0.5*v;
	for(i=0; i<l; i++)
	{
		if(y[i]== k)
			v += p*fabs(alpha[i]);
		else
			v += - alpha[i];
	}
	return v;	
}

// static void solve_l2r_npsvor(
// 	const problem *prob, double *w, const parameter *param, int k)
// {
// 	int l = prob->l;
// 	double C1 = param->C1;
// 	double C2 = param->C2;
// 	double p = param->p;
// 	int w_size = prob->n;
// 	double eps = param->eps;
// 	int i, s, iter = 0;
// 	int max_iter = 1000;
// 	int active_size = l;
// 	int *index = new int[l];
// 	double d, G;
// 	double Gmax_old = INF;
// 	double Gmax_new, Gnorm1_new;
// 	double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration
// 	double *alpha = new double[l];
// 	double *QD = new double[l];
// 	double *y = prob->y;
// 	double funval;
// 	double *obj_value = new double[max_iter];
// 	double *T = new double[max_iter];
// 	// char file_name[1024];
// 	// sprintf(file_name,"data_C1%g_C2%g_e%g_k%d.log",C1,C2,p,k);
// 	// FILE * log = fopen(file_name,"at+");
// 	clock_t start, stop;
// 	start=clock();
// 	// Initial beta can be set here. Note that
// 	memset(alpha,0,sizeof(double)*l);
// 	memset(w,0,sizeof(double)*w_size);
// 	// printf("%.3f %.3f\n",C1,C2);
//        //int nk=0;
// 	for(i=0; i<l; i++)
// 	{
// 		feature_node * const xi = prob->x[i];
// 		QD[i] = sparse_operator::nrm2_sq(xi);
// 		// sparse_operator::axpy(beta[i], xi, w);

// 		index[i] = i;
// 		//if(y[i]==k)
// 	 	//nk++;
// 	}
// 	//C2 = (double)nk/l*C2;

// 	while(iter < max_iter)
// 	{
// 		Gmax_new = 0;
// 		Gnorm1_new = 0;

// 		for(i=0; i<active_size; i++)
// 		{
// 			int j = i+rand()%(active_size-i);
// 			swap(index[i], index[j]);
// 		}

// 		for(s=0; s<active_size; s++)
// 		{
// 			i = index[s];
// 			int yki = ((y[i]>k)?1:-1);
// 			feature_node * const xi = prob->x[i];
// 			double violation = 0;
// 			if(y[i]!= k)
// 			{
// 				G = yki*sparse_operator::dot(w, xi) -1;
// 				if (alpha[i] == 0)
// 				{
// 					if (G > Gmax_old)
// 					{
// 						active_size--;
// 						swap(index[s], index[active_size]);
// 						s--;
// 						continue;
// 					}
// 					else if (G < 0)
// 						violation = -G;
// 				}
// 				else if (alpha[i] == C2)
// 				{
// 					if (G < -Gmax_old)
// 					{
// 						active_size--;
// 						swap(index[s], index[active_size]);
// 						s--;
// 						continue;
// 					}
// 					else if (G > 0)
// 						violation = G;
// 				}
// 				else
// 					violation = fabs(G);
// 				// PGmax_new = max(PGmax_new, PG);
// 				// PGmin_new = min(PGmin_new, PG);
// 				Gmax_new = max(Gmax_new, violation);
// 				Gnorm1_new += violation;

// 				if(fabs(violation) > 1.0e-12)
// 				{
// 					double alpha_old = alpha[i];
// 					alpha[i] = min(max(alpha[i] - G/QD[i], 0.0), C2);
// 					d = (alpha[i] - alpha_old)*yki;
// 					sparse_operator::axpy(d, xi, w);
// 				}
// 			} 
// 			else
// 			{
// 				G = yki*sparse_operator::dot(w, xi);
// 				double Gp = G+p;
// 				double Gn = G-p;
// 				if(alpha[i] == 0)
// 				{
// 					if(Gp < 0)
// 						violation = -Gp;
// 					else if(Gn > 0)
// 						violation = Gn;
// 					else if(Gp>Gmax_old && Gn<-Gmax_old)
// 					{
// 						active_size--;
// 						swap(index[s], index[active_size]);
// 						s--;
// 						continue;
// 					}
// 				}
// 				else if(alpha[i] >= C1)
// 				{
// 					if(Gp > 0)
// 						violation = Gp;
// 					else if(Gp < -Gmax_old)
// 					{
// 						active_size--;
// 						swap(index[s], index[active_size]);
// 						s--;
// 						continue;
// 					}
// 				}
// 				else if(alpha[i] <= -C1)
// 				{
// 					if(Gn < 0)
// 						violation = -Gn;
// 					else if(Gn > Gmax_old)
// 					{
// 						active_size--;
// 						swap(index[s], index[active_size]);
// 						s--;
// 						continue;
// 					}
// 				}
// 				else if(alpha[i] > 0)
// 					violation = fabs(Gp);
// 				else
// 					violation = fabs(Gn);

// 				Gmax_new = max(Gmax_new, violation);
// 				Gnorm1_new += violation;

// 				// obtain Newton direction d
// 				if(Gp < QD[i]*alpha[i])
// 					d = -Gp/QD[i];
// 				else if(Gn > QD[i]*alpha[i])
// 					d = -Gn/QD[i];
// 				else
// 					d = -alpha[i];

// 				if(fabs(violation) > 1.0e-12)
// 				{
// 					double alpha_old = alpha[i];
// 					alpha[i] = min(max(alpha[i]+d, -C1), C1);
// 					d = yki*(alpha[i]-alpha_old);
// 					sparse_operator::axpy(d, xi, w);
// 				}
// 			}

// 		}

// 		if(iter == 0)
// 			Gnorm1_init = Gnorm1_new;
// 		obj_value[iter] = npsvor_obj_value(prob,w,alpha,p,k);
// 		funval = obj_value[iter];
// 		stop=clock();
// 		T[iter] = (double)(stop-start)/CLOCKS_PER_SEC;
// 		// fprintf(out,"%d\t%lf\t%g\n",iter,obj_value[iter],T[iter]);
// 		iter++;
// 		// printf("%.3f %.3f %.3f  ",Gmax_old,Gnorm1_new,eps*Gnorm1_init);
// 		if(iter % 10 == 0)
// 			info(".");

// 		if(Gnorm1_new <= eps*Gnorm1_init)
// 		{
// 			if(active_size == l)
// 				break;
// 			else
// 			{
// 				active_size = l;
// 				info("*");
// 				Gmax_old = INF;
// 				continue;
// 			}
// 		}

// 		Gmax_old = Gmax_new;
// 	}
// 	info("\noptimization finished, #iter = %d\n", iter);
// 	if(iter >= max_iter)
// 		info("\nWARNING: reaching max number of iterations\nUsing -s 11 may be faster\n\n");

// 	// calculate objective value
// 	double v = 0;
// 	int nSV = 0;
// 	for(i=0; i<w_size; i++)
// 		v += w[i]*w[i];
// 	v = 0.5*v;
// 	for(i=0; i<l; i++)
// 	{
// 		if(y[i]== k)
// 			v += p*fabs(alpha[i]);
// 		else
// 			v += - alpha[i];
// 		if(alpha[i] != 0)
// 			nSV++;
// 	}

// 	// info("Objective value = %lf\n", v);
// 	// info("nSV = %d\n",nSV);
// 	// for(i=0;i<iter;i++)
// 	// 	printf("%.3f %.3f\n",(obj_value[i]-obj_value[iter-1])/fabs(obj_value[iter-1]),T[i]);
//  //    printf("\n");

// 	// sprintf(file_name,"SVs_k%d.log",k);
// 	// FILE * log = fopen(file_name,"at+");	
// 	// fprintf(log ,"%g\t%f\t%d\n",p,funval,nSV);
// 	// // fprintf(log, "\n\n");
// 	// if (NULL != log)
// 	//    fclose(log) ;	// clear the old file.		

// 	delete [] alpha;
// 	delete [] QD;
// 	delete [] index;
// }

#undef GETI
#define GETI(i) (y[i]==k?0:1)
static void solve_l2r_npsvor(
	const problem *prob, double *w, const parameter *param, int k)
{
	int l = prob->l;
	double C1 = param->C1;
	double C2 = param->C2;
	double p = param->p;
	int w_size = prob->n;
	double eps = param->eps;
	int i, s, iter = 0;
	int max_iter = 1000;
	int active_size = l;
	int *index = new int[l];
	double d, G;
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration
	double *alpha = new double[l];
	double *QD = new double[l];
	double *y = prob->y;
	// double funval;
	double T;
	function *fun_obj=NULL;
	fun_obj=new l2r_l2_npsvor_fun(prob, param, k);

	double C;
	
	double diag[2] = {0.5/C1,0.5/C2};
	double upper_bound[2] = {INF,INF};
	if(param->npsvor==1)
	{
		diag[0] = 0;
		diag[1] = 0;
		upper_bound[0] = C1;
		upper_bound[1] = C2;
	}

	// char file_name[1024];
	// sprintf(file_name,"data_C1%g_C2%g_e%g_k%d.log",C1,C2,p,k);
	// FILE * log = fopen(file_name,"at+");
	clock_t start, stop;
	start=clock();
	// Initial beta can be set here. Note that
	memset(alpha,0,sizeof(double)*l);
	memset(w,0,sizeof(double)*w_size);
	// printf("%.3f %.3f\n",C1,C2);
       //int nk=0;
	for(i=0; i<l; i++)
	{
		QD[i] = diag[GETI(i)];
		feature_node * const xi = prob->x[i];
		QD[i] += sparse_operator::nrm2_sq(xi);
		// sparse_operator::axpy(beta[i], xi, w);

		index[i] = i;
		//if(y[i]==k)
	 	//nk++;
	}
	//C2 = (double)nk/l*C2;

	while(iter < max_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;

		for(i=0; i<active_size; i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}

		for(s=0; s<active_size; s++)
		{
			i = index[s];
			int yki = ((y[i]>k)?1:-1);
			feature_node * const xi = prob->x[i];
			double violation = 0;
			if(y[i]!= k)
			{
				G = yki*sparse_operator::dot(w, xi) -1;
				C = upper_bound[GETI(i)];
				G += alpha[i]*diag[GETI(i)];				
				if (alpha[i] == 0)
				{
					if (G > Gmax_old)
					{
						active_size--;
						swap(index[s], index[active_size]);
						s--;
						continue;
					}
					else if (G < 0)
						violation = -G;
				}
				else if (alpha[i] == C)
				{
					if (G < -Gmax_old)
					{
						active_size--;
						swap(index[s], index[active_size]);
						s--;
						continue;
					}
					else if (G > 0)
						violation = G;
				}
				else
					violation = fabs(G);
				// PGmax_new = max(PGmax_new, PG);
				// PGmin_new = min(PGmin_new, PG);
				Gmax_new = max(Gmax_new, violation);
				Gnorm1_new += violation;

				if(fabs(violation) > 1.0e-12)
				{
					double alpha_old = alpha[i];
					alpha[i] = min(max(alpha[i] - G/QD[i], 0.0), C);
					d = (alpha[i] - alpha_old)*yki;
					sparse_operator::axpy(d, xi, w);
				}
			} 
			else
			{
				G = diag[GETI(i)]*alpha[i];		
				G += yki*sparse_operator::dot(w, xi);
				double Gp = G+p;
				double Gn = G-p;
				C=upper_bound[GETI(i)];

				if(alpha[i] == 0)
				{
					if(Gp < 0)
						violation = -Gp;
					else if(Gn > 0)
						violation = Gn;
					else if(Gp>Gmax_old && Gn<-Gmax_old)
					{
						active_size--;
						swap(index[s], index[active_size]);
						s--;
						continue;
					}
				}
				else if(alpha[i] >= C)
				{
					if(Gp > 0)
						violation = Gp;
					else if(Gp < -Gmax_old)
					{
						active_size--;
						swap(index[s], index[active_size]);
						s--;
						continue;
					}
				}
				else if(alpha[i] <= -C)
				{
					if(Gn < 0)
						violation = -Gn;
					else if(Gn > Gmax_old)
					{
						active_size--;
						swap(index[s], index[active_size]);
						s--;
						continue;
					}
				}
				else if(alpha[i] > 0)
					violation = fabs(Gp);
				else
					violation = fabs(Gn);

				Gmax_new = max(Gmax_new, violation);
				Gnorm1_new += violation;

				// obtain Newton direction d
				if(Gp < QD[i]*alpha[i])
					d = -Gp/QD[i];
				else if(Gn > QD[i]*alpha[i])
					d = -Gn/QD[i];
				else
					d = -alpha[i];

				if(fabs(violation) > 1.0e-12)
				{
					double alpha_old = alpha[i];
					alpha[i] = min(max(alpha[i]+d, -C), C);
					d = yki*(alpha[i]-alpha_old);
					sparse_operator::axpy(d, xi, w);
				}
			}

		}


		if(param->WithTime)
		{
			double v = 0;
			int nSV = 0;
			for(i=0; i<w_size; i++)
				v += w[i]*w[i];
			v = 0.5*v;
			for(i=0; i<l; i++)
			{
				if(y[i]== k)
					v += p*fabs(alpha[i]);
				else
					v += - alpha[i];
				if(alpha[i] != 0)
					nSV++;
			}
			double f;
			f = fun_obj->fun(w);
			// double funval = obj_value[iter];
			stop=clock();
			T = (double)(stop-start)/CLOCKS_PER_SEC;
			// fprintf(out,"%d\t%lf\t%g\n",iter,obj_value[iter],T[iter]);
			printf("%f %f %f %d\n",T,f,v,nSV);
		}


		if(iter == 0)
			Gnorm1_init = Gnorm1_new;
		iter++;
		// printf("%.3f %.3f %.3f  ",Gmax_old,Gnorm1_new,eps*Gnorm1_init);
		// if(iter % 10 == 0)
		// 	info(".");

		if(Gnorm1_new <= eps*Gnorm1_init)
		{
			if(active_size == l)
				break;
			else
			{
				active_size = l;
				// info("*");
				Gmax_old = INF;
				continue;
			}
		}

		Gmax_old = Gmax_new;
	}

	stop=clock();
	T = (double)(stop-start)/CLOCKS_PER_SEC;

	// info("\noptimization finished, #iter = %d\n", iter);
	// if(iter >= max_iter)
	// 	info("\nWARNING: reaching max number of iterations\nUsing -s 11 may be faster\n\n");

	// calculate objective value
	double v = 0;
	int nSV = 0;
	for(i=0; i<w_size; i++)
		v += w[i]*w[i];
	v = 0.5*v;
	for(i=0; i<l; i++)
	{
		if(y[i]== k)
			v += p*fabs(alpha[i]);
		else
			v += - alpha[i];
		if(alpha[i] != 0)
			nSV++;
	}

	// info("Objective value = %lf\n", v);
	// info("nSV = %d\n",nSV);
	// for(i=0;i<iter;i++)
	// 	printf("%.3f %.3f\n",(obj_value[i]-obj_value[iter-1])/fabs(obj_value[iter-1]),T[i]);
 //    printf("\n");

	// sprintf(file_name,"SVs_k%d.log",k);
	// FILE * log = fopen(file_name,"at+");	
	// fprintf(log ,"%g\t%f\t%d\n",p,funval,nSV);
	// // fprintf(log, "\n\n");
	// if (NULL != log)
	//    fclose(log) ;	// clear the old file.		

	delete [] alpha;
	delete [] QD;
	delete [] index;
}





#undef GETI
#define GETI(i) (y[i]==k?0:1)
static void solve_l2r_npsvor_Mm(
	const problem *prob, double *w, const parameter *param, int k)
{
	int l = prob->l;
	double C1 = param->C1;
	double C2 = param->C2;
	double p = param->p;
	int w_size = prob->n;
	double eps = param->eps;
	int i, s, iter = 0;
	int max_iter = 1000;
	int active_size = l;
	int *index = new int[l];
	double d, G;
	double PG;
	double PGmax_old = INF;
	double PGmin_old = -INF;
	double PGmax_new, PGmin_new;
	double *alpha = new double[l];
	double *QD = new double[l];
	double *y = prob->y;
	double T;
	function *fun_obj=NULL;
	fun_obj=new l2r_l2_npsvor_fun(prob, param, k);
	// char file_name[1024];
	// sprintf(file_name,"data_C1%g_C2%g_e%g_k%d.log",C1,C2,p,k);
	// FILE * log = fopen(file_name,"at+");
	double C;
	double diag[2] = {0.5/C1,0.5/C2};
	double upper_bound[2] = {INF,INF};
	if(param->npsvor==6)
	{
		diag[0] = 0;
		diag[1] = 0;
		upper_bound[0] = C1;
		upper_bound[1] = C2;
	}

	clock_t start, stop;
	start=clock();
	// Initial beta can be set here. Note that
	memset(alpha,0,sizeof(double)*l);
	memset(w,0,sizeof(double)*w_size);
	// printf("%.3f %.3f\n",C1,C2);
       //int nk=0;
	for(i=0; i<l; i++)
	{
		QD[i] = diag[GETI(i)];
		feature_node * const xi = prob->x[i];
		QD[i] += sparse_operator::nrm2_sq(xi);
		// sparse_operator::axpy(beta[i], xi, w);

		index[i] = i;
		//if(y[i]==k)
	 	//nk++;
	}
	//C2 = (double)nk/l*C2;

	while(iter < max_iter)
	{
		PGmax_new = -INF;
		PGmin_new = INF;

		for(i=0; i<active_size; i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}

		for(s=0; s<active_size; s++)
		{
			i = index[s];
			int yki = ((y[i]>k)?1:-1);
			feature_node * const xi = prob->x[i];
			PG = 0;	
			if(y[i]!= k)
			{
				G = yki*sparse_operator::dot(w, xi) -1;	
				C = upper_bound[GETI(i)];
				G += alpha[i]*diag[GETI(i)];						
				if (alpha[i] == 0)
				{
					if (G > PGmax_old)
					{
						active_size--;
						swap(index[s], index[active_size]);
						s--;
						continue;
					}
					else if (G < 0)
						PG = G;
				}
				else if (alpha[i] == C)
				{
					if (G < PGmin_old)
					{
						active_size--;
						swap(index[s], index[active_size]);
						s--;
						continue;
					}
					else if (G > 0)
						PG = G;
				}
				else
					PG = G;
				PGmax_new = max(PGmax_new, PG);
				PGmin_new = min(PGmin_new, PG);

				if(fabs(PG) > 1.0e-12)
				{
					double alpha_old = alpha[i];
					alpha[i] = min(max(alpha[i] - G/QD[i], 0.0), C);
					d = (alpha[i] - alpha_old)*yki;
					sparse_operator::axpy(d, xi, w);
				}
			} 
			else
			{
				G = diag[GETI(i)]*alpha[i];		
				G += yki*sparse_operator::dot(w, xi);
				double Gp = G+p;
				double Gn = G-p;
				C=upper_bound[GETI(i)];
				if(alpha[i] == 0)
				{
					if(Gp < 0)
						PG = Gp;
					else if(Gn > 0)
						PG = Gn;
					else if(Gp>PGmax_old && Gn<PGmin_old)
					{
						active_size--;
						swap(index[s], index[active_size]);
						s--;
						continue;
					}
				}
				else if(alpha[i] >=C)
				{
					if(Gp > 0)
						PG = Gp;
					else if(Gp < PGmin_old)
					{
						active_size--;
						swap(index[s], index[active_size]);
						s--;
						continue;
					}
				}
				else if(alpha[i] <= -C)
				{
					if(Gn < 0)
						PG = Gn;
					else if(Gn > PGmax_old)
					{
						active_size--;
						swap(index[s], index[active_size]);
						s--;
						continue;
					}
				}
				else if(alpha[i] > 0)
					PG = Gp;
				else
					PG = Gn;

				PGmax_new = max(PGmax_new, PG);
				PGmin_new = min(PGmin_new, PG);

				// obtain Newton direction d
				if(Gp < QD[i]*alpha[i])
					d = -Gp/QD[i];
				else if(Gn > QD[i]*alpha[i])
					d = -Gn/QD[i];
				else
					d = -alpha[i];

				if(fabs(PG) > 1.0e-12)
				{
					double alpha_old = alpha[i];
					alpha[i] = min(max(alpha[i]+d, -C), C);
					d = yki*(alpha[i]-alpha_old);
					sparse_operator::axpy(d, xi, w);
				}
			}

		}

		if(param->WithTime)
		{
			double v = 0;
			int nSV = 0;
			for(i=0; i<w_size; i++)
				v += w[i]*w[i];
			v = 0.5*v;
			for(i=0; i<l; i++)
			{
				if(y[i]== k)
					v += p*fabs(alpha[i]);
				else
					v += - alpha[i];
				if(alpha[i] != 0)
					nSV++;
			}
			double f;
			f = fun_obj->fun(w);
			// double funval = obj_value[iter];
			stop=clock();
			T = (double)(stop-start)/CLOCKS_PER_SEC;
			// fprintf(out,"%d\t%lf\t%g\n",iter,obj_value[iter],T[iter]);
			printf("%f %f %f %d\n",T,f,v,nSV);
		}

		iter++;
		// printf("%.3f %.3f %.3f  ",Gmax_old,Gnorm1_new,eps*Gnorm1_init);
		// if(iter % 10 == 0)
		// 	info(".");

		if(PGmax_new - PGmin_new <= eps)
		{
			if(active_size == l)
				break;
			else
			{
				active_size = l;
				// info("*");
				PGmax_old = INF;
				PGmin_old = -INF;
				continue;
			}
		}

		PGmax_old = PGmax_new;
		PGmin_old = PGmin_new;
		if (PGmax_old <= 0)
			PGmax_old = INF;
		if (PGmin_old >= 0)
			PGmin_old = -INF;
	}
	// info("\noptimization finished, #iter = %d\n", iter);
	// if(iter >= max_iter)
	// 	info("\nWARNING: reaching max number of iterations\nUsing -s 11 may be faster\n\n");

	// calculate objective value
	double v = 0;
	int nSV = 0;
	for(i=0; i<w_size; i++)
		v += w[i]*w[i];
	v = 0.5*v;
	for(i=0; i<l; i++)
	{
		if(y[i]== k)
			v += p*fabs(alpha[i]);
		else
			v += - alpha[i];
		if(alpha[i] != 0)
			nSV++;
	}

	// info("Objective value = %lf\n", v);
	// info("nSV = %d\n",nSV);
	// for(i=0;i<iter;i++)
	// 	printf("%.3f %.3f\n",(obj_value[i]-obj_value[iter-1])/fabs(obj_value[iter-1]),T[i]);
 //    printf("\n");

	// sprintf(file_name,"SVs_k%d.log",k);
	// FILE * log = fopen(file_name,"at+");	
	// fprintf(log ,"%g\t%f\t%d\n",p,funval,nSV);
	// // fprintf(log, "\n\n");
	// if (NULL != log)
	//    fclose(log) ;	// clear the old file.		

	delete [] alpha;
	delete [] QD;
	delete [] index;
}


static void solve_l2r_npsvor_two(
	const problem *prob, double *w, const parameter *param, double *QD)
{
	int l = prob->l;
	double C1 = param->C1;
	double C2 = param->C2;
	double p = param->p;
	int w_size = prob->n;
	double eps = param->eps;
	int i, s, iter = 0;
	int max_iter = 1000;
	int active_size = l;
	int *index = new int[l];
	// double *obj_value = new double[max_iter];
	double *T = new double[max_iter];
	double d, G;
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration
	double *alpha = new double[l];
	double *y = prob->y;
	clock_t start, stop;
	start=clock();
	// Initial beta can be set here. Note that
	memset(alpha,0,sizeof(double)*l);
	memset(w,0,sizeof(double)*w_size);
	// printf("%.3f %.3f\n",C1,C2);
    // int nk=0;
	for(i=0; i<l; i++)
		index[i] = i;


	while(iter < max_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;

		for(i=0; i<active_size; i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}

		for(s=0; s<active_size; s++)
		{
			i = index[s];
			feature_node * const xi = prob->x[i];
			double violation = 0;
			if(y[i]==1)
			{
				G = y[i]*sparse_operator::dot(w, xi) -1;
				if (alpha[i] == 0)
				{
					if (G > Gmax_old)
					{
						active_size--;
						swap(index[s], index[active_size]);
						s--;
						continue;
					}
					else if (G < 0)
						violation = -G;
				}
				else if (alpha[i] == C2)
				{
					if (G < -Gmax_old)
					{
						active_size--;
						swap(index[s], index[active_size]);
						s--;
						continue;
					}
					else if (G > 0)
						violation = G;
				}
				else
					violation = fabs(G);
				// PGmax_new = max(PGmax_new, PG);
				// PGmin_new = min(PGmin_new, PG);
				Gmax_new = max(Gmax_new, violation);
				Gnorm1_new += violation;

				if(fabs(violation) > 1.0e-12)
				{
					double alpha_old = alpha[i];
					alpha[i] = min(max(alpha[i] - G/QD[i], 0.0), C2);
					d = (alpha[i] - alpha_old)*y[i];
					sparse_operator::axpy(d, xi, w);
				}
			} 
			else
			{
				G = y[i]*sparse_operator::dot(w, xi);
				double Gp = G+p;
				double Gn = G-p;
				if(alpha[i] == 0)
				{
					if(Gp < 0)
						violation = -Gp;
					else if(Gn > 0)
						violation = Gn;
					else if(Gp>Gmax_old && Gn<-Gmax_old)
					{
						active_size--;
						swap(index[s], index[active_size]);
						s--;
						continue;
					}
				}
				else if(alpha[i] >= C1)
				{
					if(Gp > 0)
						violation = Gp;
					else if(Gp < -Gmax_old)
					{
						active_size--;
						swap(index[s], index[active_size]);
						s--;
						continue;
					}
				}
				else if(alpha[i] <= -C1)
				{
					if(Gn < 0)
						violation = -Gn;
					else if(Gn > Gmax_old)
					{
						active_size--;
						swap(index[s], index[active_size]);
						s--;
						continue;
					}
				}
				else if(alpha[i] > 0)
					violation = fabs(Gp);
				else
					violation = fabs(Gn);

				Gmax_new = max(Gmax_new, violation);
				Gnorm1_new += violation;

				// obtain Newton direction d
				if(Gp < QD[i]*alpha[i])
					d = -Gp/QD[i];
				else if(Gn > QD[i]*alpha[i])
					d = -Gn/QD[i];
				else
					d = -alpha[i];

				if(fabs(violation) > 1.0e-12)
				{
					double alpha_old = alpha[i];
					alpha[i] = min(max(alpha[i]+d, -C1), C1);
					d = y[i]*(alpha[i]-alpha_old);
					sparse_operator::axpy(d, xi, w);
				}
			}

		}

		if(iter == 0)
			Gnorm1_init = Gnorm1_new;
		// obj_value[iter] = npsvor_obj_value(prob,w,alpha,p,k);
		stop=clock();
		T[iter] = (double)(stop-start)/CLOCKS_PER_SEC;
		iter++;
		// printf("%.3f %.3f %.3f  ",Gmax_old,Gnorm1_new,eps*Gnorm1_init);
		if(iter % 10 == 0)
			info(".");

		if(Gnorm1_new <= eps*Gnorm1_init)
		{
			if(active_size == l)
				break;
			else
			{
				active_size = l;
				info("*");
				Gmax_old = INF;
				continue;
			}
		}

		Gmax_old = Gmax_new;
	}

	info("\noptimization finished, #iter = %d\n", iter);
	if(iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\nUsing -s 11 may be faster\n\n");

	// calculate objective value
	double v = 0;
	int nSV = 0;
	for(i=0; i<w_size; i++)
		v += w[i]*w[i];
	v = 0.5*v;
	for(i=0; i<l; i++)
	{
		if(y[i]== -1)
			v += p*fabs(alpha[i]);
		else
			v += - alpha[i];
		if(alpha[i] != 0)
			nSV++;
	}

	// info("Objective value = %lf\n", v);
	// info("nSV = %d\n",nSV);
	// for(i=0;i<iter;i++)
	// 	printf("%.3f %.3f\n",(obj_value[i]-obj_value[iter-1])/fabs(obj_value[iter-1]),T[i]);
 //    printf("\n");
	delete [] alpha;
	delete [] index;
}


static void sub_ramp_npsvor(
	const problem *prob, double *w, const parameter *param, double *alpha, double *QD, int k)
{
	int l = prob->l;
	double C1 = param->C1;
	double C2 = param->C2;
	double p = param->p;
	int w_size = prob->n;
	double eps = param->eps;
	int i, s, iter = 0;
	int max_iter = 1000;
	int active_size = l;
	int *index = new int[l];
	double *obj_value = new double[max_iter];
	double *T = new double[max_iter];
	double d, G;
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration
	double *y = prob->y;
	// double *w0 = new double[l];
    // memset(w0,0,sizeof(double)*w_size);		
	clock_t start, stop;
	start=clock();
	// Initial beta can be set here. Note that
	// printf("%.3f %.3f\n",C1,C2);
	// memset(w,0,sizeof(double)*w_size);
    // memset(alpha,0,sizeof(double)*l);
	for(i=0; i<l; i++)
		{
		// double yi = (double)calculate_yki(y[i], k);
		index[i] = i; 
		// sparse_operator::axpy(yi*(alpha[i]+delta[i]), prob->x[i], w);
		
		}
    
	while(iter < max_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;
        
		for(i=0; i<active_size; i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}

		for(s=0; s<active_size; s++)
		{
			i = index[s];
			int yki = calculate_yki(y[i], k);//ysk'
			feature_node * const xi = prob->x[i];
			double violation = 0;
			if(y[i]!= k)
			{
				G = yki*sparse_operator::dot(w, xi) -1;
				if (alpha[i] == 0)
				{
					if (G > Gmax_old)
					{
						active_size--;
						swap(index[s], index[active_size]);
						s--;
						continue;
					}
					else if (G < 0)
						violation = -G;
				}
				else if (alpha[i] == C2)
				{
					if (G < -Gmax_old)
					{
						active_size--;
						swap(index[s], index[active_size]);
						s--;
						continue;
					}
					else if (G > 0)
						violation = G;
				}
				else
					violation = fabs(G);
				// PGmax_new = max(PGmax_new, PG);
				// PGmin_new = min(PGmin_new, PG);
				Gmax_new = max(Gmax_new, violation);
				Gnorm1_new += violation;

				if(fabs(violation) > 1.0e-12)
				{
					double alpha_old = alpha[i];
					alpha[i] = min(max(alpha[i] - G/QD[i], 0.0), C2);
					d = (alpha[i] - alpha_old)*yki;
					sparse_operator::axpy(d, xi, w);
				}
			} 
			else
			{
				G = yki*sparse_operator::dot(w, xi);
				double Gp = G+p;
				double Gn = G-p;
				if(alpha[i] == 0)
				{
					if(Gp < 0)
						violation = -Gp;
					else if(Gn > 0)
						violation = Gn;
					else if(Gp>Gmax_old && Gn<-Gmax_old)
					{
						active_size--;
						swap(index[s], index[active_size]);
						s--;
						continue;
					}
				}
				else if(alpha[i] >= C1)
				{
					if(Gp > 0)
						violation = Gp;
					else if(Gp < -Gmax_old)
					{
						active_size--;
						swap(index[s], index[active_size]);
						s--;
						continue;
					}
				}
				else if(alpha[i] <= -C1)
				{
					if(Gn < 0)
						violation = -Gn;
					else if(Gn > Gmax_old)
					{
						active_size--;
						swap(index[s], index[active_size]);
						s--;
						continue;
					}
				}
				else if(alpha[i] > 0)
					violation = fabs(Gp);
				else
					violation = fabs(Gn);

				Gmax_new = max(Gmax_new, violation);
				Gnorm1_new += violation;

				// obtain Newton direction d
				if(Gp < QD[i]*alpha[i])
					d = -Gp/QD[i];
				else if(Gn > QD[i]*alpha[i])
					d = -Gn/QD[i];
				else
					d = -alpha[i];

				if(fabs(violation) > 1.0e-12)
				{
					double alpha_old = alpha[i];
					alpha[i] = min(max(alpha[i]+d, -C1), C1);
					d = yki*(alpha[i]-alpha_old);
					sparse_operator::axpy(d, xi, w);
				}
			}

		}

		if(iter == 0)
			Gnorm1_init = Gnorm1_new;
		obj_value[iter] = npsvor_obj_value(prob,w,alpha,p,k);
		stop=clock();
		T[iter] = (double)(stop-start)/CLOCKS_PER_SEC;
		iter++;
		// printf("%.3f %.3f %.3f  ",Gmax_old,Gnorm1_new,eps*Gnorm1_init);
		if(iter % 10 == 0)
			info(".");

		if(Gnorm1_new <= eps*Gnorm1_init)
		{
			if(active_size == l)
				break;
			else
			{
				active_size = l;
				info("*");
				Gmax_old = INF;
				continue;
			}
		}

		Gmax_old = Gmax_new;
	}

	info("\noptimization finished, #iter = %d\n", iter);
	if(iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\nUsing -s 11 may be faster\n\n");

	// calculate objective value
	double v = 0;
	int nSV = 0;
	for(i=0; i<w_size; i++)
		v += w[i]*w[i];
	v = 0.5*v;
	for(i=0; i<l; i++)
	{
		if(y[i]== k)
			v += p*fabs(alpha[i]);
		else
			v += - alpha[i];
		if(alpha[i] != 0)
			nSV++;
	}

	// info("Objective value = %lf\n", v);
	// info("nSV = %d\n",nSV);
	// for(i=0;i<iter;i++)
	// 	printf("%.3f %.3f\n",(obj_value[i]-obj_value[iter-1])/fabs(obj_value[iter-1]),T[i]);
 	//    printf("\n");
	// delete [] alpha;
	// delete [] QD;
	delete [] index;
}


static void ramp_npsvor(
	const problem *prob, double *w, const parameter *param, int k)
{
	int l = prob->l;
	double C1 = param->C1;
	double C2 = param->C2;		
	int w_size = prob->n;	
	double *alpha = new double[l];
	double *delta = new double[l];
	// double *w0= new double[w_size];

	memset(alpha,0,sizeof(double)*l);
	memset(delta,0,sizeof(double)*l);
	memset(w,0,sizeof(double)*w_size);
    
    int iter = 0, maxiter = 4;
    double *y = prob->y;
    double HB;
    double hinge_s = -1, Ins_t = 2;
	double *QD = new double[l];
	int i;

	for(i=0; i<l; i++)
		QD[i] = sparse_operator::nrm2_sq(prob->x[i]);
    
	while(iter<maxiter)
	{
		
		for(i=0; i<l; i++)
		{
			double yi =(double) calculate_yki(y[i], k);
			HB = sparse_operator::dot(w, prob->x[i]);
			if(yi*HB<hinge_s && y[i]!= k)
				delta[i] =  -C2;
			else if(HB> Ins_t && y[i]== k)
				delta[i] = -C1;
			else if(HB< -Ins_t && y[i]== k)
				delta[i] = C1;
			else delta[i] = 0;
			// info("%.1f\n ", delta[i]);
		}
		memset(w,0,sizeof(double)*w_size);
		for(i=0; i<l; i++)
		{	double yi =(double) calculate_yki(y[i], k);
			sparse_operator::axpy(yi*(alpha[i]+delta[i]), prob->x[i], w);
		}
		sub_ramp_npsvor(prob, w, param, alpha, QD, k);
		// if(r_norm<eps_pri)
		// 	break;		
		iter++;	
	}

	delete [] alpha;
	delete [] delta;
	delete [] QD;
}

double VectDot(double *u, double *v, int n)//norm2-square
{
	int j;
	double ret = 0;

	for(j=0;j<n;j++)
	{
		ret = ret + u[j]*v[j];
	}
	return ret;
}


static void Update_W(const problem *prob, const parameter *param, double *w, double *z,double *u)
{
	int w_size = prob->n;
	int l = prob->l;
	int i,iter = 0;
	double rho = param->rho;
	double alpha, beta;
	double *Xw = new double[l];
	double *r = new double[w_size];
	double *p = new double[w_size];
	double *XTXw = new double[w_size];
	double *XTXp = new double[w_size];	
	double *Xp = new double[l];		
	int max_iter = min(5,(int)sqrt(l));
	double rr;
	int inc=1;
	memset(XTXw,0,sizeof(double)*w_size);
	for(i=0;i<l;i++)
	{
		feature_node * const xi = prob->x[i];
		// sparse_operator::axpy(z[i]-u[i], xi, b);	
		Xw[i] = sparse_operator::dot(w, xi)-z[i]+u[i];	
		sparse_operator::axpy(Xw[i], xi, XTXw);		
	}
	double normr = 0;
	for(i=0;i<w_size;i++)
	{
		r[i] = w[i]+rho*XTXw[i];
		p[i] = -r[i];
		normr += fabs(r[i]);
	}

	while(normr/w_size >1e-2 && iter <= max_iter)
	{
		normr = 0;
		memset(XTXp,0,sizeof(double)*w_size);	
		for(i=0;i<l;i++)
		{
			feature_node * const xi = prob->x[i];
			Xp[i]=sparse_operator::dot(p, xi);
			sparse_operator::axpy(Xp[i], xi, XTXp);	
		}
		rr = ddot_(&w_size, r, &inc, r, &inc);
		alpha = rr/(ddot_(&w_size, p, &inc, p, &inc)
			+rho*ddot_(&w_size, p, &inc, XTXp, &inc));


		for(i=0;i<w_size;i++)
		{
			w[i] = w[i] + alpha*p[i];
			r[i] = r[i] + alpha*(p[i]+rho*XTXp[i]);
			normr += fabs(r[i]); 
		}

		beta = ddot_(&w_size, r, &inc, r, &inc)/rr; 

		for(i=0;i<w_size;i++)
		{
			p[i] = -r[i] + beta*p[i]; 
		}
		iter++;		

	}	
	// info("\nCG optimization finished, #iter = %d\n", iter);		
	delete [] Xw;
	delete [] r;
	delete [] p;
	delete [] XTXw;
	delete [] XTXp;
	delete [] Xp;			
}


int trcg(const problem *prob, const parameter *param, double *w, double *g, double *s, double *r)
{
	int i, inc = 1;
	int n = prob->n;
	double one = 1;
	double *d = new double[n];
	double *Hd = new double[n];
	double rTr, rnewTrnew, alpha, beta, cgtol;
	double eps_cg = 0.1;
	double rho = param->rho;	
	int l = prob->l;	

	for (i=0; i<n; i++)
	{
		s[i] = 0;
		r[i] = -g[i];
		d[i] = r[i];
	}
	cgtol = eps_cg*dnrm2_(&n, g, &inc);

	int cg_iter = 0;
	rTr = ddot_(&n, r, &inc, r, &inc);
	while (1)
	{
		if (dnrm2_(&n, r, &inc) <= cgtol)
			break;
		cg_iter++;
		memset(Hd,0,sizeof(double)*n);	
		for(i=0;i<l;i++)
		{
			feature_node * const xi = prob->x[i];
			double xd = sparse_operator::dot(d, xi);
			sparse_operator::axpy(xd, xi, Hd);	
		}
		dscal_(&n, &rho, Hd, &inc);		
		daxpy_(&n, &one, d, &inc, Hd, &inc);
		alpha = rTr/ddot_(&n, d, &inc, Hd, &inc);
		daxpy_(&n, &alpha, d, &inc, s, &inc);
		alpha = -alpha;
		daxpy_(&n, &alpha, Hd, &inc, r, &inc);
		rnewTrnew = ddot_(&n, r, &inc, r, &inc);
		beta = rnewTrnew/rTr;
		dscal_(&n, &beta, d, &inc);
		daxpy_(&n, &one, r, &inc, d, &inc);
		rTr = rnewTrnew;
	}

	delete[] d;
	delete[] Hd;

	return(cg_iter);
}

int ptrcg(const problem *prob, const parameter *param, double *w, double *g, double *s, double *r, double *PI)
{
	int i, inc = 1;
	int n = prob->n;
	double one = 1;
	double *d = new double[n];
	double *PId = new double[n];	
	double *Hd = new double[n];
	double rTr, rnewTrnew, alpha, beta, cgtol;
	double eps_cg = 0.1;
	double rho = param->rho;	
	int l = prob->l;	

	for (i=0; i<n; i++)
	{
		s[i] = 0;
		r[i] = -PI[i]*g[i];
		d[i] = r[i];
	}
	cgtol = eps_cg*dnrm2_(&n, r, &inc);

	int cg_iter = 0;
	rTr = ddot_(&n, r, &inc, r, &inc);
	while (1)
	{
		if (dnrm2_(&n, r, &inc) <= cgtol)
			break;
		cg_iter++;
		memset(Hd,0,sizeof(double)*n);
		for(i=0;i<n;i++)
			PId[i] = PI[i]*d[i];	
		for(i=0;i<l;i++)
		{
			feature_node * const xi = prob->x[i];
			double xd = sparse_operator::dot(PId, xi);
			sparse_operator::axpy(xd, xi, Hd);	
		}
		dscal_(&n, &rho, Hd, &inc);		
		daxpy_(&n, &one, PId, &inc, Hd, &inc);
		for(i=0;i<n;i++)
			Hd[i] = PI[i]*Hd[i];		
		alpha = rTr/ddot_(&n, d, &inc, Hd, &inc);
		daxpy_(&n, &alpha, d, &inc, s, &inc);
		alpha = -alpha;
		daxpy_(&n, &alpha, Hd, &inc, r, &inc);
		rnewTrnew = ddot_(&n, r, &inc, r, &inc);
		beta = rnewTrnew/rTr;
		dscal_(&n, &beta, d, &inc);
		daxpy_(&n, &one, r, &inc, d, &inc);
		rTr = rnewTrnew;
	}

	delete[] d;
	delete[] Hd;

	return(cg_iter);
}


static void solve_npsvor_admm(const problem *prob, double *w, const parameter *param, int k)
{
	int i,iter=0;
	int max_iter = 50;
	int tau;
	double theta, ra;
	int l = prob->l;
	int w_size = prob->n;		
	double C1 = param->C1;
	double C2 = param->C2;
	double p = param->p;
	double rho = param->rho;
	double eps = param->eps;		
	double *z = Malloc(double, l);
	double *u = Malloc(double, l);
	double s0,r0;
	double *s = new double[w_size];
	double *r = new double[w_size];
	double *g = new double[w_size];		
	double *y = prob->y;
	double u_old,one = 1;
	int cg_iter, inc=1;
	memset(z,0,sizeof(double)*l);
	memset(u,0,sizeof(double)*l);
	memset(w,0,sizeof(double)*(prob->n));
	memset(g,0,sizeof(double)*w_size);


	while(iter < max_iter)
	{

		// Update_W(prob, param, w, z, u); 
		cg_iter = trcg(prob, param, w, g, s, r);
		daxpy_(&w_size, &one, s, &inc, w, &inc);

		s0=0;r0=0;
		memset(g,0,sizeof(double)*w_size);
		for(i=0;i<l;i++)
		{	
			feature_node * const xi = prob->x[i];
			double xw = sparse_operator::dot(w, xi);
			if(y[i] != k)
			{
				ra = xw+u[i];
				double gamma = ((y[i]>k)?-1:1);
				double gra = gamma*ra;
				theta = C2/rho;
				tau = 1;
				if( gra >theta-tau)
					z[i] = ra - gamma*theta;
				else if( gra <= theta-tau &&  gra >=-tau)
					z[i] = -tau/gamma;
				else
					z[i] = ra;
			}
			else
			{
				ra = xw+u[i];
				theta = C1/rho;
				if(fabs(ra)>theta+p)
					{
						if(ra>theta+p)
							z[i] = ra - theta;
						else
							z[i] = ra + theta;
					}
				else if(fabs(ra) <= theta + p && fabs(ra)>=p)
					{	
						if(ra>=p)
							z[i] = p;
						else
							z[i] = -p;
					}
				else
					z[i] = ra;		
			}

			u_old = u[i];
			u[i] = u[i] + xw - z[i];
			r0 = max(r0,fabs(z[i]-xw));
			s0 = max(s0,fabs(u[i]-u_old));
			sparse_operator::axpy(xw-z[i]+u[i], xi, g);					
		}

		if(r0<eps || s0<eps)
			break;
		dscal_(&w_size, &rho, g, &inc);		
		daxpy_(&w_size, &one, w, &inc, g, &inc);	
		iter++;		
	}
	info("\noptimization finished, #iter = %d, cg_iter = %d\n", iter, cg_iter);	
	delete [] z;
	delete [] u;
	delete [] g;
	delete [] s;
	delete [] r;
}

static void solve_npsvor_admm_pcg(const problem *prob, double *w, const parameter *param, int k)
{
	int i,iter=0;
	int max_iter = 50;
	int tau;
	double theta, ra;
	int l = prob->l;
	int w_size = prob->n;		
	double C1 = param->C1;
	double C2 = param->C2;
	double p = param->p;
	double rho = param->rho;
	double eps = param->eps;		
	double *z = Malloc(double, l);
	double *u = Malloc(double, l);
	double *s = new double[w_size];
	double *r = new double[w_size];
	double *g = new double[w_size];
	double *P = new double[w_size];
	double *xx = new double[w_size];
	double *PI = new double[w_size];					
	double *y = prob->y;
	double one = 1;
	int cg_iter, inc=1;
	memset(z,0,sizeof(double)*l);
	memset(u,0,sizeof(double)*l);
	memset(w,0,sizeof(double)*(prob->n));
	memset(g,0,sizeof(double)*w_size);
	memset(xx,0,sizeof(double)*w_size);
	double init_gnorm=0;
	int np=0;

	for(i=0;i<l;i++)
	{
		feature_node * const xi = prob->x[i];
		sparse_operator::xxy(xi, xx);			
	}
	for(i=0;i<w_size;i++)
	{		
		P[i] = sqrt(1+rho*xx[i]);
		PI[i] = 1.0/P[i];
	}

	while(iter < max_iter)
	{

		// Update_W(prob, param, w, z, u); 
		cg_iter = ptrcg(prob, param, w, g, s, r, PI);
		for(i=0;i<w_size;i++)
			s[i] *= PI[i];
		daxpy_(&w_size, &one, s, &inc, w, &inc);

		// s0=0;r0=0;
		memset(g,0,sizeof(double)*w_size);
		for(i=0;i<l;i++)
		{	
			feature_node * const xi = prob->x[i];
			double xw = sparse_operator::dot(w, xi);
			if(y[i] != k)
			{
				ra = xw+u[i];
				double gamma = ((y[i]>k)?-1:1);
				double gra = gamma*ra;
				theta = C2/rho;
				tau = 1;
				if( gra >theta-tau)
					z[i] = ra - gamma*theta;
				else if( gra <= theta-tau &&  gra >=-tau)
					z[i] = -tau/gamma;
				else
					z[i] = ra;
			}
			else
			{
				ra = xw+u[i];
				theta = C1/rho;
				if(fabs(ra)>theta+p)
					{
						if(ra>theta+p)
							z[i] = ra - theta;
						else
							z[i] = ra + theta;
					}
				else if(fabs(ra) <= theta + p && fabs(ra)>=p)
					{	
						if(ra>=p)
							z[i] = p;
						else
							z[i] = -p;
					}
				else
					z[i] = ra;		
			}
			if(y[i]== k)
				np += 1;
			u[i] = u[i] + xw - z[i];
			sparse_operator::axpy(xw-z[i]+u[i], xi, g);					
		}
		if(iter==0)
			init_gnorm = dnrm2_(&w_size, g, &inc);
		if(dnrm2_(&w_size, g, &inc)<eps*init_gnorm)
			break;		
		dscal_(&w_size, &rho, g, &inc);		
		daxpy_(&w_size, &one, w, &inc, g, &inc);	
		iter++;		
	}
	info("\noptimization finished, #iter = %d, cg_iter = %d\n", iter, cg_iter);	
	delete [] z;
	delete [] u;
	delete [] g;
	delete [] s;
	delete [] r;
	delete [] P;
	delete [] PI;		
}



static void solve_npsvor_admm_linearApprox(const problem *prob, double *w, const parameter *param, int k)
{
	int i,iter=0;
	int w_size = prob->n;	
	int max_iter = 10;
	int tau;
	double theta, ra;
	int l = prob->l;
	double C1 = param->C1;
	double C2 = param->C2;
	double p = param->p;
	double rho = param->rho;
	double eps = param->eps;		
	double *z = Malloc(double, l);
	double *u = Malloc(double, l);
	double s,r;
	double *y = prob->y;
	double u_old;
	memset(z,0,sizeof(double)*l);
	memset(u,0,sizeof(double)*l);
	memset(w,0,sizeof(double)*(prob->n));
	double *Xw = new double[l];
	double *XTXw = new double[w_size];
	double tau0 = 0.1;
	// for(i=0;i<l;i++)
	// {
	// 	feature_node * const xi = prob->x[i];
	// 	Xw[i] = sparse_operator::dot(w, xi);	
	// }


	while(iter < max_iter)
	{
		// Update_W(prob, param, w, z, u);
		memset(XTXw,0,sizeof(double)*w_size);	
		for(i=0;i<l;i++)
		{
			feature_node * const xi = prob->x[i];
			Xw[i] = sparse_operator::dot(w, xi);
			sparse_operator::axpy(Xw[i]-z[i]+u[i], xi, XTXw);		
		}
		for(i=0;i<w_size;i++)
		{
			w[i] -= tau0*(w[i]+rho*XTXw[i]);
			if(w[i]>1)
				w[i]=1;
			else if(w[i]<-1)
				w[i]=-1;

		}
    
		s=0;r=0;
		for(i=0;i<l;i++)
		{	
			feature_node * const xi = prob->x[i];
			Xw[i] = sparse_operator::dot(w, xi);
			if(y[i] != k)
			{
				ra = Xw[i]+u[i];
				double gamma = ((y[i]>k)?-1:1);
				double gra = gamma*ra;
				theta = C2/rho;
				tau = 1;
				if( gra >theta-tau)
					z[i] = ra - gamma*theta;
				else if( gra <= theta-tau &&  gra >=-tau)
					z[i] = -tau/gamma;
				else
					z[i] = ra;
			}
			else
			{
				ra = Xw[i]+u[i];
				theta = C1/rho;
				if(fabs(ra)>theta+p)
					{
						if(ra>theta+p)
							z[i] = ra - theta;
						else
							z[i] = ra + theta;
					}
				else if(fabs(ra) <= theta + p && fabs(ra)>=p)
					{	
						if(ra>=p)
							z[i] = p;
						else
							z[i] = -p;
					}
				else
					z[i] = ra;		
			}

			u_old = u[i];
			u[i] = u[i] + Xw[i] - z[i];
			r = max(r,fabs(z[i]-Xw[i]));
			s = max(s,fabs(u[i]-u_old));
		}
		if(r<eps && s<eps)
			break;
		iter++;		
	}
	info("\noptimization finished, #iter = %d\n", iter);	
	delete [] z;
	delete [] u;
	delete [] Xw;
	delete [] XTXw;
}


static void solve_npsvor_admm_warmstart(const problem *prob, double *w, const parameter *param, int k,double *z,double *u)
{
	int i,iter=0;
	int max_iter = 10;
	int tau;
	double theta, ra;
	int l = prob->l;
	double C1 = param->C1;
	double C2 = param->C2;
	double p = param->p;
	double rho = param->rho;
	double eps = param->eps;		
	// double *z = Malloc(double, l);
	// double *u = Malloc(double, l);
	double s,r;
	double *y = prob->y;
	double u_old;
	// memset(z,0,sizeof(double)*l);
	// memset(u,0,sizeof(double)*l);
	// memset(w,0,sizeof(double)*(prob->n));

	while(iter < max_iter)
	{
		Update_W(prob, param, w, z, u); 
	    
		s=0;r=0;
		for(i=0;i<l;i++)
		{	
			feature_node * const xi = prob->x[i];
			double xw = sparse_operator::dot(w, xi);
			if(y[i] != k)
			{
				ra = xw+u[i];
				double gamma = ((y[i]>k)?-1:1);
				double gra = gamma*ra;
				theta = C2/rho;
				tau = 1;
				if( gra >theta-tau)
					z[i] = ra - gamma*theta;
				else if( gra <= theta-tau &&  gra >=-tau)
					z[i] = -tau/gamma;
				else
					z[i] = ra;
			}
			else
			{
				ra = xw+u[i];
				theta = C1/rho;
				if(fabs(ra)>theta+p)
					{
						if(ra>theta+p)
							z[i] = ra - theta;
						else
							z[i] = ra + theta;
					}
				else if(fabs(ra) <= theta + p && fabs(ra)>=p)
					{	
						if(ra>=p)
							z[i] = p;
						else
							z[i] = -p;
					}
				else
					z[i] = ra;

			}

			u_old = u[i];
			u[i] = u[i] + xw - z[i];
			r = max(r,fabs(z[i]-xw));
			s = max(s,fabs(u[i]-u_old));
		}
		if(r<eps && s<eps)
			break;
		iter++;		
	}
	info("\noptimization finished, #iter = %d\n", iter);	
	// delete [] z;
	// delete [] u;
}


/*
* ------------------modification end---------------------------
*/

// A coordinate descent algorithm for 
// L1-loss and L2-loss epsilon-SVR dual problem
//
//  min_\beta  0.5\beta^T (Q + diag(lambda)) \beta - p \sum_{i=1}^l|\beta_i| + \sum_{i=1}^l yi\beta_i,
//    s.t.      -upper_bound_i <= \beta_i <= upper_bound_i,
// 
//  where Qij = xi^T xj and
//  D is a diagonal matrix 
//
// In L1-SVM case:
// 		upper_bound_i = C
// 		lambda_i = 0
// In L2-SVM case:
// 		upper_bound_i = INF
// 		lambda_i = 1/(2*C)
//
// Given: 
// x, y, p, C
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Algorithm 4 of Ho and Lin, 2012   

#undef GETI
#define GETI(i) (0)
// To support weights for instances, use GETI(i) (i)

static void solve_l2r_l1l2_svr(
	const problem *prob, double *w, const parameter *param,
	int solver_type)
{
	int l = prob->l;
	int tl = 0;
	int active_size = l -tl;
	clock_t start, stop;
	start=clock();


	if(prob->WithTime)
	{
		tl = (int)prob->l*3/10;
		active_size = l -tl;
					
	}

	// int l = prob->l;
	double C = param->C;
	double p = param->p;
	int w_size = prob->n;
	double eps = param->eps;
	int i, s, iter = 0;
	int max_iter = 1000;
	int *index = new int[l];

	double d, G, H;
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration
	double *beta = new double[l];
	double *QD = new double[l];
	double *y = prob->y;

	// L2R_L2LOSS_SVR_DUAL
	double lambda[1], upper_bound[1];
	lambda[0] = 0.5/C;
	upper_bound[0] = INF;

	if(solver_type == L2R_L1LOSS_SVR_DUAL)
	{
		lambda[0] = 0;
		upper_bound[0] = C;
	}

	// Initial beta can be set here. Note that
	// -upper_bound <= beta[i] <= upper_bound
	for(i=0; i<l; i++)
		beta[i] = 0;

	for(i=0; i<w_size; i++)
		w[i] = 0;
	for(i=0; i<l; i++)
	{
		feature_node * const xi = prob->x[i];
		QD[i] = sparse_operator::nrm2_sq(xi);
		sparse_operator::axpy(beta[i], xi, w);

		index[i] = i;
	}


	while(iter < max_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;

		for(i=0; i<active_size; i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}

		for(s=0; s<active_size; s++)
		{
			i = index[s];
			G = -y[i] + lambda[GETI(i)]*beta[i];
			H = QD[i] + lambda[GETI(i)];

			feature_node * const xi = prob->x[i];
			G += sparse_operator::dot(w, xi);

			double Gp = G+p;
			double Gn = G-p;
			double violation = 0;
			if(beta[i] == 0)
			{
				if(Gp < 0)
					violation = -Gp;
				else if(Gn > 0)
					violation = Gn;
				else if(Gp>Gmax_old && Gn<-Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(beta[i] >= upper_bound[GETI(i)])
			{
				if(Gp > 0)
					violation = Gp;
				else if(Gp < -Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(beta[i] <= -upper_bound[GETI(i)])
			{
				if(Gn < 0)
					violation = -Gn;
				else if(Gn > Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(beta[i] > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);

			Gmax_new = max(Gmax_new, violation);
			Gnorm1_new += violation;

			// obtain Newton direction d
			if(Gp < H*beta[i])
				d = -Gp/H;
			else if(Gn > H*beta[i])
				d = -Gn/H;
			else
				d = -beta[i];

			if(fabs(d) < 1.0e-12)
				continue;

			double beta_old = beta[i];
			beta[i] = min(max(beta[i]+d, -upper_bound[GETI(i)]), upper_bound[GETI(i)]);
			d = beta[i]-beta_old;

			if(d != 0)
				sparse_operator::axpy(d, xi, w);


			if(prob->WithTime)
			{
				if(s%(active_size/5)==0)
				///measures vs time
				{

					// calculate objective value
					double v = 0;
					int nSV = 0;
					for(i=0; i<w_size; i++)
						v += w[i]*w[i];
					v = 0.5*v;
					for(i=0; i<l; i++)
					{
						v += p*fabs(beta[i]) - y[i]*beta[i] + 0.5*lambda[GETI(i)]*beta[i]*beta[i];
						if(beta[i] != 0)
							nSV++;
					}

					int nr_w = 1;
					// int t;
					double mze=0;
					double mae = 0.0, mse = 0.0, T, diff,target_label,predict_y;	
					// info("nr_class=%d", prob->nr_class);
					for(int t=l-tl;t<prob->l;t++)		
						{
							target_label = (double)prob->y[t];
							predict_y = (double)predict_label_SVR(prob->x[t], w, prob->label, nr_w, w_size, prob->nr_class);
							// info("target_label=%.3f predict_y=%.3f\n",target_label, predict_y);
							if(predict_y != target_label)
								mze += 1;
							diff = fabs(predict_y -target_label);
							mae += diff;
							mse += diff*diff;
							// printf("%d %.3f %.3f\n",mze, mae, diff);
						}				
					stop=clock();
					T = (double)(stop-start)/CLOCKS_PER_SEC;
					printf("%.4f %.3f %d %.3f %.3f %.3f\n",T, v, nSV, mze/tl, mae/tl, mse/tl);

				}
			}



		}

		if(iter == 0)
			Gnorm1_init = Gnorm1_new;
		iter++;
		// printf("%.3f %.3f %.3f  ",Gmax_old,Gnorm1_new,eps*Gnorm1_init);
		// if(iter % 10 == 0)
			// info(".");

		if(Gnorm1_new <= eps*Gnorm1_init)
		{
			if(active_size == l-tl)
				break;
			else
			{
				active_size = l-tl;
				// info("*");
				Gmax_old = INF;
				continue;
			}
		}

		Gmax_old = Gmax_new;


	}
	info("\noptimization finished, #iter = %d\n", iter);
	if(iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\nUsing -s 11 may be faster\n\n");

	// calculate objective value
	double v = 0;
	int nSV = 0;
	for(i=0; i<w_size; i++)
		v += w[i]*w[i];
	v = 0.5*v;
	for(i=0; i<l; i++)
	{
		v += p*fabs(beta[i]) - y[i]*beta[i] + 0.5*lambda[GETI(i)]*beta[i]*beta[i];
		if(beta[i] != 0)
			nSV++;
	}

	info("Objective value = %lf\n", v);
	info("nSV = %d\n",nSV);

	delete [] beta;
	delete [] QD;
	delete [] index;
}


// A coordinate descent algorithm for 
// the dual of L2-regularized logistic regression problems
//
//  min_\alpha  0.5(\alpha^T Q \alpha) + \sum \alpha_i log (\alpha_i) + (upper_bound_i - \alpha_i) log (upper_bound_i - \alpha_i),
//    s.t.      0 <= \alpha_i <= upper_bound_i,
// 
//  where Qij = yi yj xi^T xj and 
//  upper_bound_i = Cp if y_i = 1
//  upper_bound_i = Cn if y_i = -1
//
// Given: 
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Algorithm 5 of Yu et al., MLJ 2010

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

void solve_l2r_lr_dual(const problem *prob, double *w, double eps, double Cp, double Cn)
{
	int l = prob->l;
	int w_size = prob->n;
	int i, s, iter = 0;
	double *xTx = new double[l];
	int max_iter = 1000;
	int *index = new int[l];	
	double *alpha = new double[2*l]; // store alpha and C - alpha
	schar *y = new schar[l];
	int max_inner_iter = 100; // for inner Newton
	double innereps = 1e-2;
	double innereps_min = min(1e-8, eps);
	double upper_bound[3] = {Cn, 0, Cp};

	for(i=0; i<l; i++)
	{
		if(prob->y[i] > 0)
		{
			y[i] = +1;
		}
		else
		{
			y[i] = -1;
		}
	}
	
	// Initial alpha can be set here. Note that
	// 0 < alpha[i] < upper_bound[GETI(i)]
	// alpha[2*i] + alpha[2*i+1] = upper_bound[GETI(i)]
	for(i=0; i<l; i++)
	{
		alpha[2*i] = min(0.001*upper_bound[GETI(i)], 1e-8);
		alpha[2*i+1] = upper_bound[GETI(i)] - alpha[2*i];
	}

	for(i=0; i<w_size; i++)
		w[i] = 0;
	for(i=0; i<l; i++)
	{
		feature_node * const xi = prob->x[i];
		xTx[i] = sparse_operator::nrm2_sq(xi);
		sparse_operator::axpy(y[i]*alpha[2*i], xi, w);
		index[i] = i;
	}

	while (iter < max_iter)
	{
		for (i=0; i<l; i++)
		{
			int j = i+rand()%(l-i);
			swap(index[i], index[j]);
		}
		int newton_iter = 0;
		double Gmax = 0;
		for (s=0; s<l; s++)
		{
			i = index[s];
			const schar yi = y[i];
			double C = upper_bound[GETI(i)];
			double ywTx = 0, xisq = xTx[i];
			feature_node * const xi = prob->x[i];
			ywTx = yi*sparse_operator::dot(w, xi);
			double a = xisq, b = ywTx;

			// Decide to minimize g_1(z) or g_2(z)
			int ind1 = 2*i, ind2 = 2*i+1, sign = 1;
			if(0.5*a*(alpha[ind2]-alpha[ind1])+b < 0)
			{
				ind1 = 2*i+1;
				ind2 = 2*i;
				sign = -1;
			}

			//  g_t(z) = z*log(z) + (C-z)*log(C-z) + 0.5a(z-alpha_old)^2 + sign*b(z-alpha_old)
			double alpha_old = alpha[ind1];
			double z = alpha_old;
			if(C - z < 0.5 * C)
				z = 0.1*z;
			double gp = a*(z-alpha_old)+sign*b+log(z/(C-z));
			Gmax = max(Gmax, fabs(gp));

			// Newton method on the sub-problem
			const double eta = 0.1; // xi in the paper
			int inner_iter = 0;
			while (inner_iter <= max_inner_iter)
			{
				if(fabs(gp) < innereps)
					break;
				double gpp = a + C/(C-z)/z;
				double tmpz = z - gp/gpp;
				if(tmpz <= 0)
					z *= eta;
				else // tmpz in (0, C)
					z = tmpz;
				gp = a*(z-alpha_old)+sign*b+log(z/(C-z));
				newton_iter++;
				inner_iter++;
			}

			if(inner_iter > 0) // update w
			{
				alpha[ind1] = z;
				alpha[ind2] = C-z;
				sparse_operator::axpy(sign*(z-alpha_old)*yi, xi, w);
			}
		}

		iter++;
		if(iter % 10 == 0)
			info(".");

		if(Gmax < eps)
			break;

		if(newton_iter <= l/10)
			innereps = max(innereps_min, 0.1*innereps);

	}

	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\nUsing -s 0 may be faster (also see FAQ)\n\n");

	// calculate objective value

	double v = 0;
	for(i=0; i<w_size; i++)
		v += w[i] * w[i];
	v *= 0.5;
	for(i=0; i<l; i++)
		v += alpha[2*i] * log(alpha[2*i]) + alpha[2*i+1] * log(alpha[2*i+1])
			- upper_bound[GETI(i)] * log(upper_bound[GETI(i)]);
	info("Objective value = %lf\n", v);

	delete [] xTx;
	delete [] alpha;
	delete [] y;
	delete [] index;
}

// A coordinate descent algorithm for 
// L1-regularized L2-loss support vector classification
//
//  min_w \sum |wj| + C \sum max(0, 1-yi w^T xi)^2,
//
// Given: 
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Yuan et al. (2010) and appendix of LIBLINEAR paper, Fan et al. (2008)

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

static void solve_l1r_l2_svc(
	problem *prob_col, double *w, double eps,
	double Cp, double Cn)
{
	int l = prob_col->l;
	int w_size = prob_col->n;
	int j, s, iter = 0;
	int max_iter = 1000;
	int active_size = w_size;
	int max_num_linesearch = 20;

	double sigma = 0.01;
	double d, G_loss, G, H;
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration
	double d_old, d_diff;
	double loss_old, loss_new;
	double appxcond, cond;

	int *index = new int[w_size];
	schar *y = new schar[l];
	double *b = new double[l]; // b = 1-ywTx
	double *xj_sq = new double[w_size];
	feature_node *x;

	double C[3] = {Cn,0,Cp};

	// Initial w can be set here.
	for(j=0; j<w_size; j++)
		w[j] = 0;

	for(j=0; j<l; j++)
	{
		b[j] = 1;
		if(prob_col->y[j] > 0)
			y[j] = 1;
		else
			y[j] = -1;
	}
	for(j=0; j<w_size; j++)
	{
		index[j] = j;
		xj_sq[j] = 0;
		x = prob_col->x[j];
		while(x->index != -1)
		{
			int ind = x->index-1;
			x->value *= y[ind]; // x->value stores yi*xij
			double val = x->value;
			b[ind] -= w[j]*val;
			xj_sq[j] += C[GETI(ind)]*val*val;
			x++;
		}
	}

	while(iter < max_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;

		for(j=0; j<active_size; j++)
		{
			int i = j+rand()%(active_size-j);
			swap(index[i], index[j]);
		}

		for(s=0; s<active_size; s++)
		{
			j = index[s];
			G_loss = 0;
			H = 0;

			x = prob_col->x[j];
			while(x->index != -1)
			{
				int ind = x->index-1;
				if(b[ind] > 0)
				{
					double val = x->value;
					double tmp = C[GETI(ind)]*val;
					G_loss -= tmp*b[ind];
					H += tmp*val;
				}
				x++;
			}
			G_loss *= 2;

			G = G_loss;
			H *= 2;
			H = max(H, 1e-12);

			double Gp = G+1;
			double Gn = G-1;
			double violation = 0;
			if(w[j] == 0)
			{
				if(Gp < 0)
					violation = -Gp;
				else if(Gn > 0)
					violation = Gn;
				else if(Gp>Gmax_old/l && Gn<-Gmax_old/l)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(w[j] > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);

			Gmax_new = max(Gmax_new, violation);
			Gnorm1_new += violation;

			// obtain Newton direction d
			if(Gp < H*w[j])
				d = -Gp/H;
			else if(Gn > H*w[j])
				d = -Gn/H;
			else
				d = -w[j];

			if(fabs(d) < 1.0e-12)
				continue;

			double delta = fabs(w[j]+d)-fabs(w[j]) + G*d;
			d_old = 0;
			int num_linesearch;
			for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
			{
				d_diff = d_old - d;
				cond = fabs(w[j]+d)-fabs(w[j]) - sigma*delta;

				appxcond = xj_sq[j]*d*d + G_loss*d + cond;
				if(appxcond <= 0)
				{
					x = prob_col->x[j];
					sparse_operator::axpy(d_diff, x, b);
					break;
				}

				if(num_linesearch == 0)
				{
					loss_old = 0;
					loss_new = 0;
					x = prob_col->x[j];
					while(x->index != -1)
					{
						int ind = x->index-1;
						if(b[ind] > 0)
							loss_old += C[GETI(ind)]*b[ind]*b[ind];
						double b_new = b[ind] + d_diff*x->value;
						b[ind] = b_new;
						if(b_new > 0)
							loss_new += C[GETI(ind)]*b_new*b_new;
						x++;
					}
				}
				else
				{
					loss_new = 0;
					x = prob_col->x[j];
					while(x->index != -1)
					{
						int ind = x->index-1;
						double b_new = b[ind] + d_diff*x->value;
						b[ind] = b_new;
						if(b_new > 0)
							loss_new += C[GETI(ind)]*b_new*b_new;
						x++;
					}
				}

				cond = cond + loss_new - loss_old;
				if(cond <= 0)
					break;
				else
				{
					d_old = d;
					d *= 0.5;
					delta *= 0.5;
				}
			}

			w[j] += d;

			// recompute b[] if line search takes too many steps
			if(num_linesearch >= max_num_linesearch)
			{
				info("#");
				for(int i=0; i<l; i++)
					b[i] = 1;

				for(int i=0; i<w_size; i++)
				{
					if(w[i]==0) continue;
					x = prob_col->x[i];
					sparse_operator::axpy(-w[i], x, b);
				}
			}
		}

		if(iter == 0)
			Gnorm1_init = Gnorm1_new;
		iter++;
		if(iter % 10 == 0)
			info(".");

		if(Gnorm1_new <= eps*Gnorm1_init)
		{
			if(active_size == w_size)
				break;
			else
			{
				active_size = w_size;
				info("*");
				Gmax_old = INF;
				continue;
			}
		}

		Gmax_old = Gmax_new;
	}

	info("\noptimization finished, #iter = %d\n", iter);
	if(iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\n");

	// calculate objective value

	double v = 0;
	int nnz = 0;
	for(j=0; j<w_size; j++)
	{
		x = prob_col->x[j];
		while(x->index != -1)
		{
			x->value *= prob_col->y[x->index-1]; // restore x->value
			x++;
		}
		if(w[j] != 0)
		{
			v += fabs(w[j]);
			nnz++;
		}
	}
	for(j=0; j<l; j++)
		if(b[j] > 0)
			v += C[GETI(j)]*b[j]*b[j];

	info("Objective value = %lf\n", v);
	info("#nonzeros/#features = %d/%d\n", nnz, w_size);

	delete [] index;
	delete [] y;
	delete [] b;
	delete [] xj_sq;
}

// A coordinate descent algorithm for 
// L1-regularized logistic regression problems
//
//  min_w \sum |wj| + C \sum log(1+exp(-yi w^T xi)),
//
// Given: 
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Yuan et al. (2011) and appendix of LIBLINEAR paper, Fan et al. (2008)

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

static void solve_l1r_lr(
	const problem *prob_col, double *w, double eps,
	double Cp, double Cn)
{
	int l = prob_col->l;
	int w_size = prob_col->n;
	int j, s, newton_iter=0, iter=0;
	int max_newton_iter = 100;
	int max_iter = 1000;
	int max_num_linesearch = 20;
	int active_size;
	int QP_active_size;

	double nu = 1e-12;
	double inner_eps = 1;
	double sigma = 0.01;
	double w_norm, w_norm_new;
	double z, G, H;
	double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double QP_Gmax_old = INF;
	double QP_Gmax_new, QP_Gnorm1_new;
	double delta, negsum_xTd, cond;

	int *index = new int[w_size];
	schar *y = new schar[l];
	double *Hdiag = new double[w_size];
	double *Grad = new double[w_size];
	double *wpd = new double[w_size];
	double *xjneg_sum = new double[w_size];
	double *xTd = new double[l];
	double *exp_wTx = new double[l];
	double *exp_wTx_new = new double[l];
	double *tau = new double[l];
	double *D = new double[l];
	feature_node *x;

	double C[3] = {Cn,0,Cp};

	// Initial w can be set here.
	for(j=0; j<w_size; j++)
		w[j] = 0;

	for(j=0; j<l; j++)
	{
		if(prob_col->y[j] > 0)
			y[j] = 1;
		else
			y[j] = -1;

		exp_wTx[j] = 0;
	}

	w_norm = 0;
	for(j=0; j<w_size; j++)
	{
		w_norm += fabs(w[j]);
		wpd[j] = w[j];
		index[j] = j;
		xjneg_sum[j] = 0;
		x = prob_col->x[j];
		while(x->index != -1)
		{
			int ind = x->index-1;
			double val = x->value;
			exp_wTx[ind] += w[j]*val;
			if(y[ind] == -1)
				xjneg_sum[j] += C[GETI(ind)]*val;
			x++;
		}
	}
	for(j=0; j<l; j++)
	{
		exp_wTx[j] = exp(exp_wTx[j]);
		double tau_tmp = 1/(1+exp_wTx[j]);
		tau[j] = C[GETI(j)]*tau_tmp;
		D[j] = C[GETI(j)]*exp_wTx[j]*tau_tmp*tau_tmp;
	}

	while(newton_iter < max_newton_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;
		active_size = w_size;

		for(s=0; s<active_size; s++)
		{
			j = index[s];
			Hdiag[j] = nu;
			Grad[j] = 0;

			double tmp = 0;
			x = prob_col->x[j];
			while(x->index != -1)
			{
				int ind = x->index-1;
				Hdiag[j] += x->value*x->value*D[ind];
				tmp += x->value*tau[ind];
				x++;
			}
			Grad[j] = -tmp + xjneg_sum[j];

			double Gp = Grad[j]+1;
			double Gn = Grad[j]-1;
			double violation = 0;
			if(w[j] == 0)
			{
				if(Gp < 0)
					violation = -Gp;
				else if(Gn > 0)
					violation = Gn;
				//outer-level shrinking
				else if(Gp>Gmax_old/l && Gn<-Gmax_old/l)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(w[j] > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);

			Gmax_new = max(Gmax_new, violation);
			Gnorm1_new += violation;
		}

		if(newton_iter == 0)
			Gnorm1_init = Gnorm1_new;

		if(Gnorm1_new <= eps*Gnorm1_init)
			break;

		iter = 0;
		QP_Gmax_old = INF;
		QP_active_size = active_size;

		for(int i=0; i<l; i++)
			xTd[i] = 0;

		// optimize QP over wpd
		while(iter < max_iter)
		{
			QP_Gmax_new = 0;
			QP_Gnorm1_new = 0;

			for(j=0; j<QP_active_size; j++)
			{
				int i = j+rand()%(QP_active_size-j);
				swap(index[i], index[j]);
			}

			for(s=0; s<QP_active_size; s++)
			{
				j = index[s];
				H = Hdiag[j];

				x = prob_col->x[j];
				G = Grad[j] + (wpd[j]-w[j])*nu;
				while(x->index != -1)
				{
					int ind = x->index-1;
					G += x->value*D[ind]*xTd[ind];
					x++;
				}

				double Gp = G+1;
				double Gn = G-1;
				double violation = 0;
				if(wpd[j] == 0)
				{
					if(Gp < 0)
						violation = -Gp;
					else if(Gn > 0)
						violation = Gn;
					//inner-level shrinking
					else if(Gp>QP_Gmax_old/l && Gn<-QP_Gmax_old/l)
					{
						QP_active_size--;
						swap(index[s], index[QP_active_size]);
						s--;
						continue;
					}
				}
				else if(wpd[j] > 0)
					violation = fabs(Gp);
				else
					violation = fabs(Gn);

				QP_Gmax_new = max(QP_Gmax_new, violation);
				QP_Gnorm1_new += violation;

				// obtain solution of one-variable problem
				if(Gp < H*wpd[j])
					z = -Gp/H;
				else if(Gn > H*wpd[j])
					z = -Gn/H;
				else
					z = -wpd[j];

				if(fabs(z) < 1.0e-12)
					continue;
				z = min(max(z,-10.0),10.0);

				wpd[j] += z;

				x = prob_col->x[j];
				sparse_operator::axpy(z, x, xTd);
			}

			iter++;

			if(QP_Gnorm1_new <= inner_eps*Gnorm1_init)
			{
				//inner stopping
				if(QP_active_size == active_size)
					break;
				//active set reactivation
				else
				{
					QP_active_size = active_size;
					QP_Gmax_old = INF;
					continue;
				}
			}

			QP_Gmax_old = QP_Gmax_new;
		}

		if(iter >= max_iter)
			info("WARNING: reaching max number of inner iterations\n");

		delta = 0;
		w_norm_new = 0;
		for(j=0; j<w_size; j++)
		{
			delta += Grad[j]*(wpd[j]-w[j]);
			if(wpd[j] != 0)
				w_norm_new += fabs(wpd[j]);
		}
		delta += (w_norm_new-w_norm);

		negsum_xTd = 0;
		for(int i=0; i<l; i++)
			if(y[i] == -1)
				negsum_xTd += C[GETI(i)]*xTd[i];

		int num_linesearch;
		for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
		{
			cond = w_norm_new - w_norm + negsum_xTd - sigma*delta;

			for(int i=0; i<l; i++)
			{
				double exp_xTd = exp(xTd[i]);
				exp_wTx_new[i] = exp_wTx[i]*exp_xTd;
				cond += C[GETI(i)]*log((1+exp_wTx_new[i])/(exp_xTd+exp_wTx_new[i]));
			}

			if(cond <= 0)
			{
				w_norm = w_norm_new;
				for(j=0; j<w_size; j++)
					w[j] = wpd[j];
				for(int i=0; i<l; i++)
				{
					exp_wTx[i] = exp_wTx_new[i];
					double tau_tmp = 1/(1+exp_wTx[i]);
					tau[i] = C[GETI(i)]*tau_tmp;
					D[i] = C[GETI(i)]*exp_wTx[i]*tau_tmp*tau_tmp;
				}
				break;
			}
			else
			{
				w_norm_new = 0;
				for(j=0; j<w_size; j++)
				{
					wpd[j] = (w[j]+wpd[j])*0.5;
					if(wpd[j] != 0)
						w_norm_new += fabs(wpd[j]);
				}
				delta *= 0.5;
				negsum_xTd *= 0.5;
				for(int i=0; i<l; i++)
					xTd[i] *= 0.5;
			}
		}

		// Recompute some info due to too many line search steps
		if(num_linesearch >= max_num_linesearch)
		{
			for(int i=0; i<l; i++)
				exp_wTx[i] = 0;

			for(int i=0; i<w_size; i++)
			{
				if(w[i]==0) continue;
				x = prob_col->x[i];
				sparse_operator::axpy(w[i], x, exp_wTx);
			}

			for(int i=0; i<l; i++)
				exp_wTx[i] = exp(exp_wTx[i]);
		}

		if(iter == 1)
			inner_eps *= 0.25;

		newton_iter++;
		Gmax_old = Gmax_new;

		info("iter %3d  #CD cycles %d\n", newton_iter, iter);
	}

	info("=========================\n");
	info("optimization finished, #iter = %d\n", newton_iter);
	if(newton_iter >= max_newton_iter)
		info("WARNING: reaching max number of iterations\n");

	// calculate objective value

	double v = 0;
	int nnz = 0;
	for(j=0; j<w_size; j++)
		if(w[j] != 0)
		{
			v += fabs(w[j]);
			nnz++;
		}
	for(j=0; j<l; j++)
		if(y[j] == 1)
			v += C[GETI(j)]*log(1+1/exp_wTx[j]);
		else
			v += C[GETI(j)]*log(1+exp_wTx[j]);

	info("Objective value = %lf\n", v);
	info("#nonzeros/#features = %d/%d\n", nnz, w_size);

	delete [] index;
	delete [] y;
	delete [] Hdiag;
	delete [] Grad;
	delete [] wpd;
	delete [] xjneg_sum;
	delete [] xTd;
	delete [] exp_wTx;
	delete [] exp_wTx_new;
	delete [] tau;
	delete [] D;
}

// transpose matrix X from row format to column format
static void transpose(const problem *prob, feature_node **x_space_ret, problem *prob_col)
{
	int i;
	int l = prob->l;
	int n = prob->n;
	size_t nnz = 0;
	size_t *col_ptr = new size_t [n+1];
	feature_node *x_space;
	prob_col->l = l;
	prob_col->n = n;
	prob_col->y = new double[l];
	prob_col->x = new feature_node*[n];

	for(i=0; i<l; i++)
		prob_col->y[i] = prob->y[i];

	for(i=0; i<n+1; i++)
		col_ptr[i] = 0;
	for(i=0; i<l; i++)
	{
		feature_node *x = prob->x[i];
		while(x->index != -1)
		{
			nnz++;
			col_ptr[x->index]++;
			x++;
		}
	}
	for(i=1; i<n+1; i++)
		col_ptr[i] += col_ptr[i-1] + 1;

	x_space = new feature_node[nnz+n];
	for(i=0; i<n; i++)
		prob_col->x[i] = &x_space[col_ptr[i]];

	for(i=0; i<l; i++)
	{
		feature_node *x = prob->x[i];
		while(x->index != -1)
		{
			int ind = x->index-1;
			x_space[col_ptr[ind]].index = i+1; // starts from 1
			x_space[col_ptr[ind]].value = x->value;
			col_ptr[ind]++;
			x++;
		}
	}
	for(i=0; i<n; i++)
		x_space[col_ptr[i]].index = -1;

	*x_space_ret = x_space;

	delete [] col_ptr;
}

// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
static void group_classes(const problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
{
	int l = prob->l;
	int max_nr_class = 16;
	int nr_class = 0;
	int *label = Malloc(int,max_nr_class);
	int *count = Malloc(int,max_nr_class);
	int *data_label = Malloc(int,l);
	int i;

	for(i=0;i<l;i++)
	{
		int this_label = (int)prob->y[i];
		int j;
		for(j=0;j<nr_class;j++)
		{
			if(this_label == label[j])
			{
				++count[j];
				break;
			}
		}
		data_label[i] = j;
		if(j == nr_class)
		{
			if(nr_class == max_nr_class)
			{
				max_nr_class *= 2;
				label = (int *)realloc(label,max_nr_class*sizeof(int));
				count = (int *)realloc(count,max_nr_class*sizeof(int));
			}
			label[nr_class] = this_label;
			count[nr_class] = 1;
			++nr_class;
		}
	}

	//
	// Labels are ordered by their first occurrence in the training set. 
	// However, for two-class sets with -1/+1 labels and -1 appears first, 
	// we swap labels to ensure that internally the binary SVM has positive data corresponding to the +1 instances.
	//
	if (nr_class == 2 && label[0] == -1 && label[1] == 1)
	{
		swap(label[0],label[1]);
		swap(count[0],count[1]);
		for(i=0;i<l;i++)
		{
			if(data_label[i] == 0)
				data_label[i] = 1;
			else
				data_label[i] = 0;
		}
	}

	int *start = Malloc(int,nr_class);
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];
	for(i=0;i<l;i++)
	{
		perm[start[data_label[i]]] = i;
		++start[data_label[i]];
	}
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];

	*nr_class_ret = nr_class;
	*label_ret = label;
	*start_ret = start;
	*count_ret = count;
	free(data_label);
}

/*My modification begin*/
// static void train_one_svor(const problem *prob, const parameter *param, double *w, double *b, double C)
// {
// 	//inner and outer tolerances for TRON
// 	double eps = param->eps;
// 	double wl = param->wl;
// 	//clock_t start, stop;
// 	//start=clock();
// 	solve_l2r_svor(prob, w, b, eps, C, wl)
// 	//solve_l2r_svor(&sub_prob, param, model_->w, model_->b, param->C,nr_class);
// }
/*My modification end*/


static void train_one(const problem *prob, const parameter *param, double *w, double Cp, double Cn)
{
	//inner and outer tolerances for TRON
	double eps = param->eps;
	double eps_cg = 0.1;
	if(param->init_sol != NULL)
		eps_cg = 0.5;

	int pos = 0;
	int neg = 0;
	for(int i=0;i<prob->l;i++)
		if(prob->y[i] > 0)
			pos++;
	neg = prob->l - pos;
	double primal_solver_tol = eps*max(min(pos,neg), 1)/prob->l;

	function *fun_obj=NULL;
/*
* -------------------my modification begin---------------------------
*/
	// double para_rho = param->rho;
	// double para_wl = param->wl;
	//clock_t start, stop;
	//start=clock();
/*
* -------------------my modification end---------------------------
*/
	//fprintf(stderr, "My test0\n");
	switch(param->solver_type)
	{
/*
* -------------------my modification begin---------------------------
*/
// 		case L2R_SVOR:
// 		{
// 			// //fprintf(stderr, "My test1\n");
// 			// if(Cp != Cn)//Cp and Cn are the same in this case Cp = Cn = C
// 			// 	fprintf(stderr, "ERROR: Cp and Cn should be the same in this case\n");
// 			solve_l2r_svor(prob, w, b, eps, C, para_wl)
// 			break; 
// 		}
 /*
* -------------------my modification end---------------------------
*/
		case L2R_LR:
		{
			double *C = new double[prob->l];
			for(int i = 0; i < prob->l; i++)
			{
				if(prob->y[i] > 0)
					C[i] = Cp;
				else
					C[i] = Cn;
			}
			fun_obj=new l2r_lr_fun(prob, C);
			TRON tron_obj(fun_obj, primal_solver_tol, eps_cg);
			tron_obj.set_print_string(liblinear_print_string);
			tron_obj.tron(w);
			delete fun_obj;
			delete[] C;
			break;
		}
		case L2R_L2LOSS_SVC:
		{
			double *C = new double[prob->l];
			for(int i = 0; i < prob->l; i++)
			{
				if(prob->y[i] > 0)
					C[i] = Cp;
				else
					C[i] = Cn;
			}
			fun_obj=new l2r_l2_svc_fun(prob, C);
			TRON tron_obj(fun_obj, primal_solver_tol, eps_cg);
			tron_obj.set_print_string(liblinear_print_string);
			tron_obj.tron(w);
			delete fun_obj;
			delete[] C;
			break;
		}
		case L2R_L2LOSS_SVC_DUAL:
			solve_l2r_l1l2_svc(prob, w, eps, Cp, Cn, L2R_L2LOSS_SVC_DUAL);
			break;
		case L2R_L1LOSS_SVC_DUAL:
			solve_l2r_l1l2_svc(prob, w, eps, Cp, Cn, L2R_L1LOSS_SVC_DUAL);
			//solve_l2r_l1l2_svmop(prob, w, eps, Cp, Cn, L2R_L1LOSS_SVC_DUAL);
			break;
		case L2R_SVMOP:
			if(param->npsvor==1)
				solve_l2r_l1l2_svc(prob, w, eps, Cp, Cn, L2R_L1LOSS_SVC_DUAL);
			else if(param->npsvor==2)
				solve_l2r_l1l2_svmop(prob, w, eps, Cp, Cn, L2R_L2LOSS_SVC_DUAL);
			break;
		case L2R_NPSVOR:
			//solve_l2r_l1l2_svc(prob, w, eps, Cp, Cn, L2R_L1LOSS_SVC_DUAL);
			solve_l2r_l1l2_svmop(prob, w, eps, Cp, Cn, L2R_L1LOSS_SVC_DUAL);
			break;			
		case L1R_L2LOSS_SVC:
		{
			problem prob_col;
			feature_node *x_space = NULL;
			transpose(prob, &x_space ,&prob_col);
			solve_l1r_l2_svc(&prob_col, w, primal_solver_tol, Cp, Cn);
			delete [] prob_col.y;
			delete [] prob_col.x;
			delete [] x_space;
			break;
		}
		case L1R_LR:
		{
			problem prob_col;
			feature_node *x_space = NULL;
			transpose(prob, &x_space ,&prob_col);
			solve_l1r_lr(&prob_col, w, primal_solver_tol, Cp, Cn);
			delete [] prob_col.y;
			delete [] prob_col.x;
			delete [] x_space;
			break;
		}
		case L2R_LR_DUAL:
			solve_l2r_lr_dual(prob, w, eps, Cp, Cn);
			break;
		case L2R_L2LOSS_SVR:
		{
			double *C = new double[prob->l];
			for(int i = 0; i < prob->l; i++)
				C[i] = param->C;

			fun_obj=new l2r_l2_svr_fun(prob, C, param->p);
			TRON tron_obj(fun_obj, param->eps);
			tron_obj.set_print_string(liblinear_print_string);
			tron_obj.tron(w);
			delete fun_obj;
			delete[] C;
			break;

		}
		case L2R_L1LOSS_SVR_DUAL:
			solve_l2r_l1l2_svr(prob, w, param, L2R_L1LOSS_SVR_DUAL);
			break;
		case L2R_L2LOSS_SVR_DUAL:
			solve_l2r_l1l2_svr(prob, w, param, L2R_L2LOSS_SVR_DUAL);
			break;
		default:
			fprintf(stderr, "ERROR: unknown solver_type\n");
			break;
	}
/*
* -------------------my modification end---------------------------
*/
  	//stop=clock();
	//printf("Training Time:%f seconds.\n", (double)(stop-start)/CLOCKS_PER_SEC);
/*
* -------------------my modification end---------------------------
*/
}

// Calculate the initial C for parameter selection
static double calc_start_C(const problem *prob, const parameter *param)
{
	int i;
	double xTx,max_xTx;
	max_xTx = 0;
	for(i=0; i<prob->l; i++)
	{
		xTx = 0;
		feature_node *xi=prob->x[i];
		while(xi->index != -1)
		{
			double val = xi->value;
			xTx += val*val;
			xi++;
		}
		if(xTx > max_xTx)
			max_xTx = xTx;
	}

	double min_C = 1.0;
	if(param->solver_type == L2R_LR)
		min_C = 1.0 / (prob->l * max_xTx);
	else if(param->solver_type == L2R_L2LOSS_SVC)
		min_C = 1.0 / (2 * prob->l * max_xTx);

	return pow( 2, floor(log(min_C) / log(2.0)) );
}


//
// Interface functions
//
model* train(const problem *prob, const parameter *param)
{
	int i,j;
	int l = prob->l;
	int n = prob->n;
	int w_size = prob->n;
	model *model_ = Malloc(model,1);

	if(prob->bias>=0)
		model_->nr_feature=n-1;
	else
		model_->nr_feature=n;
	model_->param = *param;
	model_->bias = prob->bias;

	if(check_regression_model(model_))
	{
		int nr_class;
		int *label = NULL;
		int *start = NULL;
		int *count = NULL;
		int *perm = Malloc(int,l);		
		model_->w = Malloc(double, w_size);
		group_classes(prob,&nr_class,&label,&start,&count,perm);
		for(i=0; i<w_size; i++)
			model_->w[i] = 0;
		model_->nr_class = nr_class;
		model_->label = Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
			model_->label[i] = label[i];
		for(i=0;i<nr_class-1;i++)
			for(j=i+1;j< nr_class;j++)
				if(model_->label[i] > model_->label[j])
					swap(model_->label[i],model_->label[j]); 

		problem sub_prob;
		sub_prob.label = Malloc(double,nr_class);
		for(i=0;i<nr_class;i++)
			  sub_prob.label[i] = model_->label[i]; 
		sub_prob.WithTime = param->WithTime;
	
		sub_prob.l = l;
		sub_prob.n = n;
		sub_prob.x = Malloc(feature_node *,sub_prob.l);
		sub_prob.y = Malloc(double,sub_prob.l);
		sub_prob.nr_class = nr_class;

		int *index = new int[sub_prob.l];
		for(i=0; i<l; i++)
		{
			index[i]=i;			
		}
		for(i=0; i<l; i++)
		{
			int j = i+rand()%(l-i);
			swap(index[i], index[j]);
		}

		if(model_->param.npsvor==2)
        {
			for(int k=0; k<l; k++)
				{
					sub_prob.x[k] = prob->x[index[k]];
					sub_prob.y[k] = (double)(prob->y[index[k]]-1)/(nr_class -1);
				}	        	

		}
		else
		{
			for(int k=0; k<l; k++)
				{
					sub_prob.x[k] = prob->x[index[k]];
					sub_prob.y[k] = prob->y[index[k]];
				}		
		}	


		train_one(&sub_prob, param, model_->w, 0, 0);
	}
	else
	{
		int nr_class;
		int *label = NULL;
		int *start = NULL;
		int *count = NULL;
		int *perm = Malloc(int,l);

		// group training data of the same class
		group_classes(prob,&nr_class,&label,&start,&count,perm);

		model_->nr_class=nr_class;
		model_->label = Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
			model_->label[i] = label[i];
		// calculate weighted C
		double *weighted_C = Malloc(double, nr_class);
		for(i=0;i<nr_class;i++)
			weighted_C[i] = param->C;
		for(i=0;i<param->nr_weight;i++)
		{
			for(j=0;j<nr_class;j++)
				if(param->weight_label[i] == label[j])
					break;
			if(j == nr_class)
				fprintf(stderr,"WARNING: class label %d specified in weight is not found\n", param->weight_label[i]);
			else
				weighted_C[j] *= param->weight[i];
		}

		// constructing the subproblem
		// // constructing the subproblem
		int *index = new int[l];
		for(i=0; i<l; i++)
		{
			index[i]=i;			
		}
		for(i=0; i<l; i++)
		{
			int j = i+rand()%(l-i);
			swap(index[i], index[j]);
		}


		feature_node **x = Malloc(feature_node *,l);
		for(i=0;i<l;i++)
			x[i] = prob->x[index[i]];

		int k;
		problem sub_prob;
		sub_prob.l = l;
		sub_prob.n = n;
		sub_prob.x = Malloc(feature_node *,sub_prob.l);
		sub_prob.y = Malloc(double,sub_prob.l);

		for(k=0; k<sub_prob.l; k++)
			sub_prob.x[k] = x[k];



		// multi-class svm by Crammer and Singer
		if(param->solver_type == MCSVM_CS)
		{
			for(i=0;i<nr_class-1;i++)
				for(j=i+1;j< nr_class;j++)
					if(model_->label[i] > model_->label[j])
						swap(model_->label[i],model_->label[j]); 

			sub_prob.label = Malloc(double,nr_class);
			for(i=0;i<nr_class;i++)
				  sub_prob.label[i] = model_->label[i];    
			sub_prob.WithTime = param->WithTime;



			if(param->npsvor==7)  //NPSSVOR
			{
				for(i=0; i< l; i++)
					sub_prob.y[i] = prob->y[index[i]]-1;				
				model_->w=Malloc(double, n*nr_class);		
				Solver_NPSSVOR Solver(&sub_prob, nr_class, param->C1, param->C2, param->p, param->eps);
				Solver.Solve(model_->w);
				info("NPSSVOR\n\n");
				// for(i=0;i<n*nr_class;i++)
				// 	printf("%d %.2f ",i,model_->w[i]);

			}
			else if(param->npsvor==6)
			{
				for(i=0; i< l; i++)
					sub_prob.y[i] = prob->y[index[i]]-1;			
				model_->w=Malloc(double, n*nr_class);		
				Solver_CSOR_all Solver(&sub_prob, nr_class, param->C, param->eps);
				Solver.Solve(model_->w);
				info("CSOR all SDM\n\n");
				// for(i=0;i<n*nr_class;i++)
				// 	printf("%d %.2f ",i,model_->w[i]);

			}
			else if(param->npsvor==5)
			{
				for(i=0; i< l; i++)
					sub_prob.y[i] = prob->y[index[i]]-1;				
				model_->w=Malloc(double, n*nr_class);		
				solve_SNPSVOR(&sub_prob, nr_class, model_->w, param);
				info("SNPSVOR\n\n");
				// for(i=0;i<n*nr_class;i++)
				// 	printf("%d %.2f ",i,model_->w[i]);

			}
			else if(param->npsvor==4)
			{
				for(i=0; i< l; i++)
					sub_prob.y[i] = prob->y[index[i]]-1;				
				model_->w=Malloc(double, n*nr_class);		
				solve_SSVOR(&sub_prob, nr_class, model_->w, param->eps, param->C);
				info("SSVOR DCD\n\n");
				// for(i=0;i<n*nr_class;i++)
				// 	printf("%d %.2f ",i,model_->w[i]);

			}
			else if(param->npsvor==3)
			{
				for(i=0; i< l; i++)
					sub_prob.y[i] = prob->y[index[i]]-1;				
				model_->w=Malloc(double, n*nr_class);		
				Solver_SSVOR Solver(&sub_prob, nr_class, param->C, param->eps);
				Solver.Solve(model_->w);
				info("SSVOR\n\n");
				// for(i=0;i<n*nr_class;i++)
				// 	printf("%d %.2f ",i,model_->w[i]);

			}
			else if(param->npsvor==2)
			{
				for(i=0; i< l; i++)
					sub_prob.y[i] = prob->y[index[i]]-1;				
				model_->w=Malloc(double, n*nr_class);		
				Solver_CSOR Solver(&sub_prob, nr_class, param->C, param->eps);
				Solver.Solve(model_->w);
				info("CSOR SDM\n\n");
				// for(i=0;i<n*nr_class;i++)
				// 	printf("%d %.2f ",i,model_->w[i]);

			}
			else
			{
				model_->w=Malloc(double, n*nr_class);
				for(i=0; i< l; i++)
					sub_prob.y[i] = prob->y[index[i]]-1;				
				Solver_MCSVM_CS Solver(&sub_prob, nr_class, weighted_C, param->eps);
				Solver.Solve(model_->w);
			}
		}
		else if(nr_class == 2 && model_->param.solver_type!= L2R_SVOR && 
		model_->param.solver_type != L2R_NPSVOR && model_->param.solver_type!= L2R_SVMOP)    /*My modification*/
			{   
				model_->w=Malloc(double, w_size);
				int e0 = start[0]+count[0];
				k=0;
				for(; k<e0; k++)
					sub_prob.y[index[i]] = +1;
				for(; k<sub_prob.l; k++)
					sub_prob.y[k] = -1;
				
				if(param->init_sol != NULL)
					for(i=0;i<w_size;i++)
						model_->w[i] = param->init_sol[i];
				else
					for(i=0;i<w_size;i++)
						model_->w[i] = 0;

				train_one(&sub_prob, param, model_->w, weighted_C[0], weighted_C[1]);
				free(sub_prob.y);				
			}
			else if(model_->param.solver_type== L2R_SVOR) //My modification begin
				{
				// if(nr_class == 2)
				// 	for(i=0;i<prob->l;i++)
				// 		if(subprob->y[i]== model_->label[1])
				// 			subprob->y[i]==1;
				// 		else
				// 			subprob->y[i]= 2;
				info("%d %d\n",model_->label[0],model_->label[1]);
				model_->w=Malloc(double, w_size);
				model_->b=Malloc(double, nr_class-1); //add
				for(i=0; i< l; i++)
					sub_prob.y[i] = prob->y[index[i]];
				for(i=0;i<nr_class-1;i++) //add
					model_->b[i] = 0; //add
				if(param->init_sol != NULL)
					{for(i=0;i<w_size;i++)
						model_->w[i] = param->init_sol[i];
					}
				else
					{
					for(i=0;i<w_size;i++)
						model_->w[i] = 0;
					}
					info("%g\n",param->svor);
				for(i=0;i<nr_class-1;i++)
					for(j=i+1;j< nr_class;j++)
						if(model_->label[i] > model_->label[j])
							swap(model_->label[i],model_->label[j]); 			
				if(param->svor==1){
					solve_l2r_svor(&sub_prob, param, model_->w, model_->b, model_->label, nr_class);
					}
				else if(param->svor==2)
					{solve_l2r_svor_full(&sub_prob, param, model_->w, model_->b, model_->label, nr_class);}

				for(i=0;i<nr_class-1;i++) 
					info("b %.6f\n",model_->b[i]);										
				} // My modification end
			else if (model_->param.solver_type== L2R_NPSVOR)
			{	
				for(i=0; i< l; i++)
					sub_prob.y[i] = prob->y[index[i]];				
				for(i=0;i<nr_class-1;i++)
					for(j=i+1;j< nr_class;j++)
						if(model_->label[i] > model_->label[j])
							swap(model_->label[i],model_->label[j]); 

				model_->w=Malloc(double, w_size*nr_class);
				double *w=Malloc(double, w_size);
				double *z = Malloc(double, l);
				double *u = Malloc(double, l);
				memset(z,0,sizeof(double)*l);
				memset(u,0,sizeof(double)*l);

				for(i=0;i<nr_class;i++)
				{
					// if(param->init_sol != NULL)
					// 	for(j=0;j<w_size;j++)
					// 		w[j] = param->init_sol[j*nr_class+i];
					// else
					// 	for(j=0;j<w_size;j++)
					// 		w[j] = 0;
					if(param->npsvor==1){
						printf("\n%d\n",model_->label[i]);
						solve_l2r_npsvor(&sub_prob, w, param, model_->label[i]);
						//ramp_npsvor(&sub_prob, w, param, model_->label[i]);
						}
					else if(param->npsvor==2)
					{
						printf("\n%d\n",model_->label[i]);
						solve_l2r_npsvor(&sub_prob, w, param, model_->label[i]);
						// ramp_npsvor(&sub_prob, w, param, model_->label[i]);
					}							
					else if(param->npsvor==0)
					{
						printf("\n%d\n",model_->label[i]);	
						int nk = 0;
						for(j=0;j<sub_prob.l;j++) 
							if(sub_prob.y[j]== model_->label[i])
								nk++;
						solve_l2r_npsvor_full(&sub_prob, w, param, model_->label[i],nk);
					}
					else if(param->npsvor==8)
					{
						printf("\n%d\n",model_->label[i]);	
						int nk = 0;
						for(j=0;j<sub_prob.l;j++) 
							if(sub_prob.y[j]== model_->label[i])
								nk++;
						solve_l2r_npsvor_full_Mm(&sub_prob, w, param, model_->label[i],nk);
					}
					else if(param->npsvor==3)
					{
						double eps = param->eps;
						double eps_cg = 0.1;
						if(param->init_sol != NULL)
							eps_cg = 0.5;

						int pos = 0;
						int neg = 0;
						for(j =0;j<prob->l;j++)
							if(prob->y[j] == model_->label[i])
								pos++;
						neg = prob->l - pos;
						double primal_solver_tol = eps*max(min(pos,neg), 1)/prob->l;						
						function *fun_obj=NULL;
						fun_obj=new l2r_l2_npsvor_fun(prob, param, model_->label[i]);
						TRON tron_obj(fun_obj, primal_solver_tol, eps_cg, param->WithTime);
						tron_obj.set_print_string(liblinear_print_string);
						tron_obj.ptron(w);
						delete fun_obj;
					}						
					else if(param->npsvor==4)
					{
						printf("\n%d\n",model_->label[i]);
						double eps = param->eps;
						double eps_cg = 0.1;
						if(param->init_sol != NULL)
							eps_cg = 0.5;

						int pos = 0;
						int neg = 0;
						for(j =0;j<prob->l;j++)
							if(prob->y[j] == model_->label[i])
								pos++;
						neg = prob->l - pos;
						double primal_solver_tol = eps*max(min(pos,neg), 1)/prob->l;						
						function *fun_obj=NULL;
						fun_obj=new l2r_l2_npsvor_fun(prob, param, model_->label[i]);
						TRON tron_obj(fun_obj, primal_solver_tol, eps_cg, param->WithTime);
						tron_obj.set_print_string(liblinear_print_string);
						tron_obj.tron(w);
						delete fun_obj;
					}
					else if(param->npsvor==6)  //L1 NPSVOR
					{
						printf("\n%d\n",model_->label[i]);	
						solve_l2r_npsvor_Mm(&sub_prob, w, param, model_->label[i]);
					}					
					else if(param->npsvor==7)  //L2 NPSVOR
					{
						    solve_l2r_npsvor_Mm(&sub_prob, w, param, model_->label[i]);
						    //ramp_npsvor(&sub_prob, w, param, model_->label[i]);
					}
					else if(param->npsvor==11)
					{
						//solve_l2r_npsvor(&sub_prob, w, param, model_->label[i]);
						double *w1 = new double[w_size];
						double *w2 = new double[w_size];
						double *QD = new double[l];
						memset(w1,0,sizeof(double)*w_size);
						memset(w2,0,sizeof(double)*w_size);
						memset(QD,0,sizeof(double)*l);
						if(i<nr_class-1)
						{
							for(j=0; j<l; j++)
							{
								feature_node * const xi = sub_prob.x[j];
								QD[j] = sparse_operator::nrm2_sq(xi);
							}							
							for(j=0; j< l; j++)
								if(prob->y[perm[j]]>model_->label[i]) 
									sub_prob.y[j] = 1;
								else sub_prob.y[j] = -1;						
							solve_l2r_npsvor_two(&sub_prob, w1, param,QD);
							for(j=0; j< l; j++)
								if(prob->y[perm[j]] > model_->label[i]) 
									sub_prob.y[j] = -1;
								else sub_prob.y[j] = 1;						
							solve_l2r_npsvor_two(&sub_prob, w2, param,QD);
						}
						for(j=0;j<w_size;j++)
							w[j] = w1[j]-w2[j];
					}	
					else if(param->npsvor==5)
					{
						if(i==0 ||i==nr_class-1)
						{						
							for(j=0; j< l; j++)
								if(prob->y[perm[j]]<=model_->label[i]) 
									sub_prob.y[j] = -1;
								else sub_prob.y[j] = 1;
							train_one(&sub_prob, param, w, param->C1, param->C2);						

						}
						else
						{
							for(j=0; j< l; j++)
								sub_prob.y[j] = prob->y[perm[j]];
							solve_l2r_npsvor(&sub_prob, w, param, model_->label[i]);
						}
					}
					else if(param->npsvor==8){
						solve_npsvor_admm(&sub_prob, w, param, model_->label[i]);
						}
					else if(param->npsvor==9){
						solve_npsvor_admm_pcg(&sub_prob, w, param, model_->label[i]);
						}						
					else if(param->npsvor==10){
						solve_npsvor_admm_warmstart(&sub_prob, w, param, model_->label[i], z, u);
						}
					else if(param->npsvor==11){
						solve_npsvor_admm_linearApprox(&sub_prob, w, param, model_->label[i]);
						}
					else if(param->npsvor==11){
						ramp_npsvor(&sub_prob, w, param, model_->label[i]);
						}						
					for(int j=0;j<w_size;j++)
						model_->w[j*nr_class+i] = w[j];					
				}
				if(param->npsvor!=4 && param->npsvor!=3)
						free(w);
				free(sub_prob.y);
				free(z);
				free(u);

								
			}
			else if (model_->param.solver_type== L2R_SVMOP)
			{
				// for(i=0; i< l; i++)
				// 	sub_prob.y[i] = prob->y[perm[i]];					
				for(i=0;i<nr_class-1;i++)
					for(j=i+1;j< nr_class;j++)
						if(model_->label[i] > model_->label[j])
							swap(model_->label[i],model_->label[j]);				
				model_->w=Malloc(double, w_size*(nr_class-1));
				double *w=Malloc(double, w_size);
				for(i=0;i<nr_class-1;i++)
				{

					for(int s=0;s<prob->l;s++)
						if(prob->y[index[s]]> model_->label[i])
							sub_prob.y[s]=1;
						else
							sub_prob.y[s]= -1;

					if(param->init_sol != NULL)
						for(j=0;j<w_size;j++)
							w[j] = param->init_sol[j*(nr_class-1)+i];
					else
						for(j=0;j<w_size;j++)
							w[j] = 0;
					train_one(&sub_prob, param, w, weighted_C[i], param->C);

					for(int j=0;j<w_size;j++)
						{
							model_->w[j*(nr_class-1)+i] = w[j];
						}
				}
				free(w);
				free(sub_prob.y);		
			}							
			else
			{
				for(i=0;i<nr_class-1;i++)
					for(j=i+1;j< nr_class;j++)
						if(model_->label[i] > model_->label[j])
							swap(model_->label[i],model_->label[j]);				
				model_->w=Malloc(double, w_size*nr_class);
				double *w=Malloc(double, w_size);
				for(i=0;i<nr_class;i++)
				{
					for(int s=0;s<prob->l;s++)
						if(prob->y[index[s]]== model_->label[i])
							sub_prob.y[s]=1;
						else
							sub_prob.y[s]= -1;

					if(param->init_sol != NULL)
						for(j=0;j<w_size;j++)
							w[j] = param->init_sol[j*nr_class+i];
					else
						for(j=0;j<w_size;j++)
							w[j] = 0;

					train_one(&sub_prob, param, w, weighted_C[i], param->C);

					for(int j=0;j<w_size;j++)
						model_->w[j*nr_class+i] = w[j];
				}
				free(w);
				free(sub_prob.y);
			}
       // info("sssssss%.6f\n",model_->w[1]);
		free(x);
		free(label);
		free(start);
		free(count);
		free(perm);
		free(sub_prob.x);
		// free(sub_prob.y);
		free(weighted_C);
	}
	// info("sssssss%.6f\n",model_->w[1]);
	// info("b %.6f\n",model_->b[0]);		
	return model_;
}

void cross_validation(const problem *prob, const parameter *param, int nr_fold, double *target)
{
	// printf("%g\n",param->C);
	int i;
	int *fold_start;	
	int l = prob->l;
	int *perm = Malloc(int,l);
	if (nr_fold > l)
	{
		nr_fold = l;
		fprintf(stderr,"WARNING: # folds > # data. Will use # folds = # data instead (i.e., leave-one-out cross validation)\n");
	}
	fold_start = Malloc(int,nr_fold+1);
	for(i=0;i<l;i++) perm[i]=i;
	for(i=0;i<l;i++)
	{
		int j = i+rand()%(l-i);
		swap(perm[i],perm[j]);
	}
	for(i=0;i<=nr_fold;i++)
		fold_start[i]=i*l/nr_fold;

	for(i=0;i<nr_fold;i++)
	{
		int begin = fold_start[i];
		int end = fold_start[i+1];
		int j,k;
		struct problem subprob;

		subprob.bias = prob->bias;
		subprob.n = prob->n;
		subprob.l = l-(end-begin);
		subprob.x = Malloc(struct feature_node*,subprob.l);
		subprob.y = Malloc(double,subprob.l);

		k=0;
		for(j=0;j<begin;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		for(j=end;j<l;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		struct model *submodel = train(&subprob,param);
		for(j=begin;j<end;j++)
			target[perm[j]] = predict(submodel,prob->x[perm[j]]);
		free_and_destroy_model(&submodel);
		free(subprob.x);
		free(subprob.y);
	}
	free(fold_start);
	free(perm);
}

static const char *solver_type_table[]=
{
	"L2R_LR", "L2R_L2LOSS_SVC_DUAL", "L2R_L2LOSS_SVC", "L2R_L1LOSS_SVC_DUAL", "MCSVM_CS",
	"L1R_L2LOSS_SVC", "L1R_LR", "L2R_LR_DUAL",
	"L2R_SVOR", "L2R_NPSVOR", "L2R_SVMOP",//My modification
	"L2R_L2LOSS_SVR", "L2R_L2LOSS_SVR_DUAL", "L2R_L1LOSS_SVR_DUAL", NULL
};

void find_parameter_C(const problem *prob, const parameter *param, int nr_fold, double start_C, double max_C, double *best_C, 
	double *best_acc_rate, double *best_mae_rate, const char *input_file_name)
{
	// variables for CV
	int i;
	int *fold_start;
	int l = prob->l;
	int *perm = Malloc(int, l);
	double *target = Malloc(double, prob->l);
	struct problem *subprob = Malloc(problem,nr_fold);

	clock_t start, stop;
	double cvtime;
	char file_name[1024];
	sprintf(file_name,"%s_%s%g_cvdetials.log",solver_type_table[param->solver_type],input_file_name,prob->bias);
	FILE * out = fopen(file_name,"w+t");

	// variables for warm start
	double ratio = 2;
	double **prev_w = Malloc(double*, nr_fold);
	for(i = 0; i < nr_fold; i++)
		prev_w[i] = NULL;
	int num_unchanged_w = 0;
	struct parameter param1 = *param;
	void (*default_print_string) (const char *) = liblinear_print_string;

	if (nr_fold > l)
	{
		nr_fold = l;
		fprintf(stderr,"WARNING: # folds > # data. Will use # folds = # data instead (i.e., leave-one-out cross validation)\n");
	}
	fold_start = Malloc(int,nr_fold+1);
	for(i=0;i<l;i++) perm[i]=i;
	for(i=0;i<l;i++)
	{
		int j = i+rand()%(l-i);
		swap(perm[i],perm[j]);
	}
	for(i=0;i<=nr_fold;i++)
		fold_start[i]=i*l/nr_fold;

	for(i=0;i<nr_fold;i++)
	{
		int begin = fold_start[i];
		int end = fold_start[i+1];
		int j,k;

		subprob[i].bias = prob->bias;
		subprob[i].n = prob->n;
		subprob[i].l = l-(end-begin);
		subprob[i].x = Malloc(struct feature_node*,subprob[i].l);
		subprob[i].y = Malloc(double,subprob[i].l);

		k=0;
		for(j=0;j<begin;j++)
		{
			subprob[i].x[k] = prob->x[perm[j]];
			subprob[i].y[k] = prob->y[perm[j]];
			++k;
		}
		for(j=end;j<l;j++)
		{
			subprob[i].x[k] = prob->x[perm[j]];
			subprob[i].y[k] = prob->y[perm[j]];
			++k;
		}

	}

	*best_acc_rate = 0;
	*best_mae_rate = INF;
	if(start_C <= 0)
		start_C = calc_start_C(prob,param);
	param1.C = start_C;

	while(param1.C <= max_C)
	{
		start=clock();
		//Output disabled for running CV at a particular C
		set_print_string_function(&print_null);
		if(param->solver_type == L2R_NPSVOR || param->solver_type == MCSVM_CS)
		{
			param1.C1 = param1.C;
			param1.C2 = param1.C;	
		}		

		for(i=0; i<nr_fold; i++)
		{
			int j;
			int begin = fold_start[i];
			int end = fold_start[i+1];

			param1.init_sol = prev_w[i];
			struct model *submodel = train(&subprob[i],&param1);
			int total_w_size;
			if((submodel->nr_class == 2 && submodel->param.solver_type != L2R_NPSVOR)||submodel->param.solver_type == L2R_SVOR||check_regression_model(submodel))
				total_w_size = subprob[i].n;
			else if(submodel->param.solver_type == L2R_SVMOP)
				total_w_size = subprob[i].n * (submodel->nr_class-1);
			else
				total_w_size = subprob[i].n * submodel->nr_class;
				
			if(prev_w[i] == NULL)
			{
				prev_w[i] = Malloc(double, total_w_size);
				for(j=0; j<total_w_size; j++)
					prev_w[i][j] = submodel->w[j];
			}
			else if(num_unchanged_w >= 0)
			{
				double norm_w_diff = 0;
				for(j=0; j<total_w_size; j++)
				{
					norm_w_diff += (submodel->w[j] - prev_w[i][j])*(submodel->w[j] - prev_w[i][j]);
					prev_w[i][j] = submodel->w[j];
				}
				norm_w_diff = sqrt(norm_w_diff);

				if(norm_w_diff > 1e-15)
					num_unchanged_w = -1;
			}
			else
			{
				for(j=0; j<total_w_size; j++)
					prev_w[i][j] = submodel->w[j];
			}

			for(j=begin; j<end; j++)
				target[perm[j]] = predict(submodel,prob->x[perm[j]]);

			free_and_destroy_model(&submodel);
		}
		set_print_string_function(default_print_string);

		int total_correct = 0;
		double total_abserror=0;
		double total_sqerror=0;
		for(i=0; i<prob->l; i++)
			{if(target[i] == prob->y[i])
				++total_correct;
			 total_abserror += fabs(target[i] - prob->y[i]);
			 total_sqerror += fabs(target[i] - prob->y[i])*fabs(target[i] - prob->y[i]);
			}

		double current_mae_rate = (double)total_abserror/prob->l;
		double current_mse_rate = (double)total_sqerror/prob->l;
		double current_acc_rate = (double)total_correct/prob->l;
		if(current_mae_rate < *best_mae_rate)
		{
			*best_C = param1.C;
			*best_mae_rate = current_mae_rate;
			*best_acc_rate = current_acc_rate;
		}

		// info("log2c=%7.2f\tacc_rate=%.6f\tmae_rate=%.6f\tmse_rate=%.6f\n",log(param1.C)/log(2.0),current_acc_rate,current_mae_rate,current_mse_rate );


		stop=clock();
		cvtime = (double)(stop-start)/CLOCKS_PER_SEC;
		info("%7.2f\t%.6f\t%.6f\t%.6f\t%g\n",log(param1.C)/log(2.0),current_acc_rate,current_mae_rate,current_mse_rate,cvtime );
		num_unchanged_w++;
		if(num_unchanged_w == 3)
			break;	
		fprintf(out, "%7.2f\t%.6f\t%.6f\t%.6f\t%g\n",log(param1.C)/log(2.0),current_acc_rate,current_mae_rate,current_mse_rate,cvtime);	    
		param1.C = param1.C*ratio;	
	}
	// if(param1.C > max_C && max_C > start_C) 
	// 	info("warning: maximum C reached.\n");

	if (NULL != out)
	   fclose(out) ;	// clear the old file.	 

	free(fold_start);
	free(perm);
	free(target);
	for(i=0; i<nr_fold; i++)
	{
		free(subprob[i].x);
		free(subprob[i].y);
		free(prev_w[i]);
	}
	free(prev_w);
	free(subprob);
}


void find_parameter_npsvor(const problem *prob, const parameter *param, int nr_fold, 
	double start_C, double max_C, double *best_C1, double *best_C2, double *best_acc_rate,
	 double *best_mae_rate, const char *input_file_name)
{
	// variables for CV
	int i;
	int *fold_start;
	int l = prob->l;
	int *perm = Malloc(int, l);
	double *target = Malloc(double, prob->l);
	struct problem *subprob = Malloc(problem,nr_fold);

	clock_t start, stop;
	double cvtime;
	char file_name[1024];
	sprintf(file_name,"%s_%s%gc1c2_cvdetials.log",solver_type_table[param->solver_type],input_file_name,prob->bias);
	FILE * out = fopen(file_name,"w+t");

	// variables for warm start
	double ratio = 2;
	struct parameter param1 = *param;
	void (*default_print_string) (const char *) = liblinear_print_string;

	if (nr_fold > l)
	{
		nr_fold = l;
		fprintf(stderr,"WARNING: # folds > # data. Will use # folds = # data instead (i.e., leave-one-out cross validation)\n");
	}
	fold_start = Malloc(int,nr_fold+1);
	for(i=0;i<l;i++) perm[i]=i;
	for(i=0;i<l;i++)
	{
		int j = i+rand()%(l-i);
		swap(perm[i],perm[j]);
	}
	for(i=0;i<=nr_fold;i++)
		fold_start[i]=i*l/nr_fold;

	for(i=0;i<nr_fold;i++)
	{
		int begin = fold_start[i];
		int end = fold_start[i+1];
		int j,k;

		subprob[i].bias = prob->bias;
		subprob[i].n = prob->n;
		subprob[i].l = l-(end-begin);
		subprob[i].x = Malloc(struct feature_node*,subprob[i].l);
		subprob[i].y = Malloc(double,subprob[i].l);

		k=0;
		for(j=0;j<begin;j++)
		{
			subprob[i].x[k] = prob->x[perm[j]];
			subprob[i].y[k] = prob->y[perm[j]];
			++k;
		}
		for(j=end;j<l;j++)
		{
			subprob[i].x[k] = prob->x[perm[j]];
			subprob[i].y[k] = prob->y[perm[j]];
			++k;
		}

	}

	*best_acc_rate = 0;
	*best_mae_rate = INF;
	if(start_C <= 0)
		start_C = calc_start_C(prob,param);
	param1.C1 = param1.C;
	while(param1.C1 <= max_C)
	{
	    param1.C2 = param1.C;		
		while(param1.C2 <= max_C)
		{
		start=clock();			
		//Output disabled for running CV at a particular C
		set_print_string_function(&print_null);

		for(i=0; i<nr_fold; i++)
		{
			int j;
			int begin = fold_start[i];
			int end = fold_start[i+1];

			param1.init_sol = NULL;
			struct model *submodel = train(&subprob[i],&param1);
			for(j=begin; j<end; j++)
				target[perm[j]] = predict(submodel,prob->x[perm[j]]);			
			free_and_destroy_model(&submodel);
		}
		set_print_string_function(default_print_string);

		int total_correct = 0;
		double total_abserror=0;
		for(i=0; i<prob->l; i++)
			{if(target[i] == prob->y[i])
				++total_correct;
			 total_abserror += fabs(target[i] - prob->y[i]);
			}

		double current_mae_rate = (double)total_abserror/prob->l;
		double current_acc_rate = (double)total_correct/prob->l;
		if(current_mae_rate < *best_mae_rate)
		{
			*best_C1 = param1.C1;
			*best_C2 = param1.C2;
			*best_mae_rate = current_mae_rate;
			*best_acc_rate = current_acc_rate;
		}
		info("log2c=%7.2f\tlog2c=%7.2f\tacc_rate=%.6f\tmae_rate=%.6f\n",log(param1.C1)/log(2.0),log(param1.C2)/log(2.0),current_acc_rate,current_mae_rate);
		stop=clock();
		cvtime = (double)(stop-start)/CLOCKS_PER_SEC;	
		fprintf(out, "%7.2f\t%7.2f\t%.6f\t%.6f\t%g\n",log(param1.C1)/log(2.0),log(param1.C2)/log(2.0),current_acc_rate,current_mae_rate,cvtime);	    	
		param1.C2 = param1.C2*ratio;			
		}
		param1.C1 = param1.C1*ratio;
	}
	
	if((param1.C1 > max_C||param1.C2 > max_C) && max_C > start_C) 
		info("warning: maximum C reached.\n");
	if (NULL != out)
	   fclose(out) ;	// clear the old file.	 

	free(fold_start);
	free(perm);
	free(target);
	for(i=0; i<nr_fold; i++)
	{
		free(subprob[i].x);
		free(subprob[i].y);
	}
	free(subprob);
}


// void find_parameter_npsvor(const problem *prob, const parameter *param, int nr_fold, 
// 	double start_C, double max_C, double *best_C1, double *best_C2, double *best_acc_rate,
// 	 double *best_mae_rate, const char *input_file_name)
// {
// 	// variables for CV
// 	int i;
// 	int *fold_start;
// 	int l = prob->l;
// 	int *perm = Malloc(int, l);
// 	double *target = Malloc(double, prob->l);
// 	struct problem *subprob = Malloc(problem,nr_fold);

// 	clock_t start, stop;
// 	double cvtime;
// 	char file_name[1024];
// 	sprintf(file_name,"%s_%s%gc1c2_cvdetials.log",solver_type_table[param->solver_type],input_file_name,prob->bias);
// 	FILE * out = fopen(file_name,"w+t");

// 	// variables for warm start
// 	double ratio = 2;
// 	double **prev_w = Malloc(double*, nr_fold);
// 	for(i = 0; i < nr_fold; i++)
// 		prev_w[i] = NULL;
// 	struct parameter param1 = *param;
// 	void (*default_print_string) (const char *) = liblinear_print_string;

// 	if (nr_fold > l)
// 	{
// 		nr_fold = l;
// 		fprintf(stderr,"WARNING: # folds > # data. Will use # folds = # data instead (i.e., leave-one-out cross validation)\n");
// 	}
// 	fold_start = Malloc(int,nr_fold+1);
// 	for(i=0;i<l;i++) perm[i]=i;
// 	for(i=0;i<l;i++)
// 	{
// 		int j = i+rand()%(l-i);
// 		swap(perm[i],perm[j]);
// 	}
// 	for(i=0;i<=nr_fold;i++)
// 		fold_start[i]=i*l/nr_fold;

// 	for(i=0;i<nr_fold;i++)
// 	{
// 		int begin = fold_start[i];
// 		int end = fold_start[i+1];
// 		int j,k;

// 		subprob[i].bias = prob->bias;
// 		subprob[i].n = prob->n;
// 		subprob[i].l = l-(end-begin);
// 		subprob[i].x = Malloc(struct feature_node*,subprob[i].l);
// 		subprob[i].y = Malloc(double,subprob[i].l);

// 		k=0;
// 		for(j=0;j<begin;j++)
// 		{
// 			subprob[i].x[k] = prob->x[perm[j]];
// 			subprob[i].y[k] = prob->y[perm[j]];
// 			++k;
// 		}
// 		for(j=end;j<l;j++)
// 		{
// 			subprob[i].x[k] = prob->x[perm[j]];
// 			subprob[i].y[k] = prob->y[perm[j]];
// 			++k;
// 		}

// 	}

// 	*best_acc_rate = 0;
// 	*best_mae_rate = INF;
// 	if(start_C <= 0)
// 		start_C = calc_start_C(prob,param);
// 	param1.C1 = param1.C;
// 	while(param1.C1 <= max_C)
// 	{
// 	    param1.C2 = param1.C;		
// 		while(param1.C2 <= max_C)
// 		{
// 		int num_unchanged_w = 0;	
// 		start=clock();			
// 		//Output disabled for running CV at a particular C
// 		set_print_string_function(&print_null);

// 		for(i=0; i<nr_fold; i++)
// 		{
// 			int j;
// 			int begin = fold_start[i];
// 			int end = fold_start[i+1];

// 			param1.init_sol = prev_w[i];
// 			struct model *submodel = train(&subprob[i],&param1);

// 			int total_w_size;
// 			total_w_size = subprob[i].n * submodel->nr_class;

// 			if(prev_w[i] == NULL)
// 			{
// 				prev_w[i] = Malloc(double, total_w_size);
// 				for(j=0; j<total_w_size; j++)
// 					prev_w[i][j] = submodel->w[j];
// 			}
// 			else if(num_unchanged_w >= 0)
// 			{
// 				double norm_w_diff = 0;
// 				for(j=0; j<total_w_size; j++)
// 				{
// 					norm_w_diff += (submodel->w[j] - prev_w[i][j])*(submodel->w[j] - prev_w[i][j]);
// 					prev_w[i][j] = submodel->w[j];
// 				}
// 				norm_w_diff = sqrt(norm_w_diff);

// 				if(norm_w_diff > 1e-15)
// 					num_unchanged_w = -1;
// 			}
// 			else
// 			{
// 				for(j=0; j<total_w_size; j++)
// 					prev_w[i][j] = submodel->w[j];
// 			}
// 			for(j=begin; j<end; j++)
// 				target[perm[j]] = predict(submodel,prob->x[perm[j]]);			
// 			free_and_destroy_model(&submodel);
// 		}
// 		set_print_string_function(default_print_string);

// 		int total_correct = 0;
// 		double total_abserror=0;
// 		for(i=0; i<prob->l; i++)
// 			{if(target[i] == prob->y[i])
// 				++total_correct;
// 			 total_abserror += fabs(target[i] - prob->y[i]);
// 			}

// 		double current_mae_rate = (double)total_abserror/prob->l;
// 		double current_acc_rate = (double)total_correct/prob->l;
// 		if(current_mae_rate < *best_mae_rate)
// 		{
// 			*best_C1 = param1.C1;
// 			*best_C2 = param1.C2;
// 			*best_mae_rate = current_mae_rate;
// 			*best_acc_rate = current_acc_rate;
// 		}
// 		info("log2c=%7.2f\tlog2c=%7.2f\tacc_rate=%.6f\tmae_rate=%.6f\n",log(param1.C1)/log(2.0),log(param1.C2)/log(2.0),current_acc_rate,current_mae_rate);
// 		num_unchanged_w++;
// 		stop=clock();
// 		cvtime = (double)(stop-start)/CLOCKS_PER_SEC;	
// 		fprintf(out, "%7.2f\t%7.2f\t%.6f\t%.6f\t%g\n",log(param1.C1)/log(2.0),log(param1.C2)/log(2.0),current_acc_rate,current_mae_rate,cvtime);	    	
// 		param1.C2 = param1.C2*ratio;
// 		if(num_unchanged_w == 3)
// 			break;				

// 		}
// 		param1.C1 = param1.C1*ratio;
// 	}
	
// 	if((param1.C1 > max_C||param1.C2 > max_C) && max_C > start_C) 
// 		info("warning: maximum C reached.\n");
// 	if (NULL != out)
// 	   fclose(out) ;	// clear the old file.	 

// 	free(fold_start);
// 	free(perm);
// 	free(target);
// 	for(i=0; i<nr_fold; i++)
// 	{
// 		free(subprob[i].x);
// 		free(subprob[i].y);
// 		free(prev_w[i]);
// 	}
// 	free(prev_w);
// 	free(subprob);
// }



double predict_values(const struct model *model_, const struct feature_node *x, double *dec_values)
{
	int idx;
	int n;
	if(model_->bias>=0)
		n=model_->nr_feature+1;
	else
		n=model_->nr_feature;
	double *w=model_->w;
	int nr_class=model_->nr_class;
	int i;
	int nr_w;
	if((nr_class==2 && model_->param.solver_type != MCSVM_CS && 
		model_->param.solver_type != L2R_NPSVOR)||model_->param.solver_type == L2R_SVOR||check_regression_model(model_))
		nr_w = 1;
	else if(model_->param.solver_type == L2R_SVMOP)
		nr_w = nr_class-1;
	else
		nr_w = nr_class;

	const feature_node *lx=x;
	for(i=0;i<nr_w;i++)
		dec_values[i] = 0;
	for(; (idx=lx->index)!=-1; lx++)
	{
		// the dimension of testing data may exceed that of training
			if(idx<=n)
				for(i=0;i<nr_w;i++)
					dec_values[i] += w[(idx-1)*nr_w+i]*lx->value;
	}

	if(nr_class==2 && model_->param.solver_type!= L2R_SVOR && model_->param.solver_type != L2R_NPSVOR
		&& model_->param.solver_type != L2R_SVMOP)
	{
		if(check_regression_model(model_))
			return (fabs(dec_values[0]-model_->label[0])<fabs(dec_values[0]-model_->label[1]))?model_->label[0]:model_->label[1];
		else
			return (dec_values[0]>0)?model_->label[0]:model_->label[1];
	}
	else if (model_->param.solver_type == L2R_SVOR)// My modification begin
	{
        if(model_->param.npsvor==1)
        {
			int count = 1;
				for(int j=0;j<nr_class-1;j++)
				{
					if (dec_values[0] + model_->b[j]>0)
						count = count +1;
				}
				return model_->label[count-1];   	
        }
        else 
 		{
			int count = 0;
				for(int j=0;j<nr_class-1;j++)
				{
					if (dec_values[0] + model_->b[j]>0)
						count  = j+1;
				}
			return model_->label[count];	 			
 		}       			
	}                   //  My modification end
	else if (model_->param.solver_type == L2R_NPSVOR && model_->param.npsvor!=5)
	{
  //******************************************predict method 1		
		// int dec_min_idx = 0;
		// for(i=1;i<nr_class;i++)
		// {
		// 	if(fabs(dec_values[i]) < fabs(dec_values[dec_min_idx]))
		// 		dec_min_idx = i;
		// }
		// return model_->label[dec_min_idx];
  //******************************************predict method 2
		int count = 1;
			for(int j=0;j<nr_class-1;j++)
			{
				// if (dec_values[j+1]+dec_values[j]>0)
				// 	{count  += 1;}
				if (dec_values[j+1]+dec_values[j]>0)
					count += 1;

			}
		return model_->label[count-1];	


		// int count = 0;
		// for(int j=0;j<nr_class-1;j++)
		// {
		// 	if (dec_values[j+1]+dec_values[j]>0)
		// 		count  = j+1;
		// }
		// return model_->label[count];      



		// int count = 1;
		// 	for(int j=0;j<nr_class-1;j++)
		// 	{
		// 		// if (dec_values[j+1]+dec_values[j]>0)
		// 		// 	{count  += 1;}
		// 		if (dec_values[j+1]+dec_values[j]>0)
		// 			count = max(count,j+2);

		// 	}
		// return model_->label[count-1];					
  //******************************************predict method 3
		// double *dec = Malloc(double, nr_class*(nr_class+1)/2);
		// int *vote = Malloc(int,nr_class);
		// for(i=0;i<nr_class;i++)
		// 	vote[i] = 0;

		// int p=0;
		// for(i=0;i<nr_class;i++)
		// 	for(int j=i+1;j<nr_class;j++)
		// 	{
		// 		dec[p] = dec_values[j] + dec_values[i];

		// 		if(dec[p] > 0)
		// 			++vote[j];
		// 		else
		// 			++vote[i];
		// 		p++;
		// 	}

		// int vote_max_idx = 0;
		// for(i=1;i<nr_class;i++)
		// 	if(vote[i] > vote[vote_max_idx])
		// 		vote_max_idx = i;

		// free(vote);
		// return model_->label[vote_max_idx];

	}
	else if (model_->param.solver_type == L2R_NPSVOR && model_->param.npsvor==5)
	{
		int count = 1;
			for(int j=0;j<nr_class-1;j++)
			{
				if (dec_values[j+1]+dec_values[j]>0 && (j> 0 && j< nr_class-2))
					count += 1;
				else if((dec_values[0]>0 && j==0) || (dec_values[nr_class-1]>0 && j ==nr_class-2))
					count += 1;
			}
		return model_->label[count-1];	
	}
	else if(check_regression_model(model_))
	{


        if(model_->param.npsvor==2)
        {
	 		int dec_min_idx = 0;
			double td = fabs(dec_values[0]-(double)(model_->label[0]-1)/(nr_class-1));
			for(i=1;i<nr_class;i++)
			{
				double ti = fabs(dec_values[0]-(double)(model_->label[i]-1)/(nr_class-1));
				if( ti < td)
					{
						dec_min_idx = i;
						td = ti;
					}
			}
			return model_->label[dec_min_idx];    


        }
        else
        {
 			int dec_min_idx = 0;
			double td = fabs(dec_values[0]-model_->label[0]);
			for(i=1;i<nr_class;i++)
			{
				double ti = fabs(dec_values[0]-model_->label[i]);
				if( ti < td)
					{
						dec_min_idx = i;
						td = ti;
					}
			}
			return model_->label[dec_min_idx];	   	
        }

	}
	else if (model_->param.solver_type == L2R_SVMOP || (model_->param.solver_type == L2R_NPSVOR && model_->param.npsvor==4))
	{
		// int dec_max_idx = 0;
		// double *prob_estimates = Malloc(double, model_->nr_class);
		// for(i=0;i<nr_class-1;i++)
		// 	dec_values[i]=1/(1+exp(-dec_values[i]));
		// // double sum=0;
		// // for(i=0; i<nr_class-1; i++)
		// // 	sum+=dec_values[i];

		// // for(i=0; i<nr_class-1; i++)
		// // 	dec_values[i]=dec_values[i]/sum;

	
		// for(i=0;i<nr_class;i++)
		// {			
		// 	if(i==0) 
		// 		prob_estimates[i] = 1-dec_values[0];
		// 	else if(i==nr_class-1) 
		// 		prob_estimates[i] = dec_values[nr_class-1];
		// 	else
		// 		prob_estimates[i] = dec_values[i-1]-dec_values[i];
		// }
		// // for(i=0;i<nr_class-1;i++)
		// // 	{info("%.3f ",dec_values[i]);}
		// for(i=1;i<nr_class;i++)
		// {
		// 	if(prob_estimates[i] > prob_estimates[dec_max_idx])
		// 		dec_max_idx = i;
		// }
		// return model_->label[dec_max_idx];	

  //******************************************predict method 3
		// int dec_max_idx = 0;
		// for(i=0;i<nr_w;i++)
		// 	dec_values[i]=1/(1+exp(-dec_values[i]));
		// double *prob_estimates = Malloc(double, model_->nr_class);
		// if(nr_class==2) // for binary classification
		// {
		// 	prob_estimates[0]=1.- dec_values[0]; 
		//     prob_estimates[1] = dec_values[0];
		// }
		// else
		// {		
		// 	double sum=0;

		// 	for(i=0; i<nr_w; i++)
		// 		sum += dec_values[i];
		// 	for(i=0; i<nr_w; i++)
		// 		dec_values[i]=dec_values[i]/sum;
		//  	// info("%.3f ", dec_values[0]);		
		// 	for(i=0;i<nr_class;i++)
		// 	{
		// 		if(i==0) 
		// 			prob_estimates[i] = 1-dec_values[0];
		// 		else if(i==nr_class-1) 
		// 			prob_estimates[i] = dec_values[nr_w-1];
		// 		else
		// 			prob_estimates[i] = dec_values[i-1] - dec_values[i];
		// 	}	
		// }	
		// // for(i=0;i<nr_class;i++)
		// // {info("%.3f ", prob_estimates[i]);}
		// for(i=1;i<nr_class;i++)
		// {
		// 	if(prob_estimates[i] > prob_estimates[dec_max_idx])
		// 		dec_max_idx = i;
		// }
		// return model_->label[dec_max_idx];	

  //******************************************predict method 3

		if(model_->param.npsvor==3)
		{
			int dec_max_idx = 0;
			double *prob_estimates = Malloc(double, model_->nr_class);
			for(i=0;i<nr_class-1;i++)
				dec_values[i]=1/(1+exp(-dec_values[i]));
			// double sum=0;
			// for(i=0; i<nr_class-1; i++)
			// 	sum+=dec_values[i];

			// for(i=0; i<nr_class-1; i++)
			// 	dec_values[i]=dec_values[i]/sum;

		
			for(i=0;i<nr_class;i++)
			{			
				if(i==0) 
					prob_estimates[i] = 1-dec_values[0];
				else if(i==nr_class-1) 
					prob_estimates[i] = dec_values[nr_class-1];
				else
					prob_estimates[i] = dec_values[i-1]*(1-dec_values[i]);
			}
			// for(i=0;i<nr_class-1;i++)
			// 	{info("%.3f ",dec_values[i]);}
			for(i=1;i<nr_class;i++)
			{
				if(prob_estimates[i] > prob_estimates[dec_max_idx])
					dec_max_idx = i;
			}
			return model_->label[dec_max_idx];				
		}
        else if(model_->param.npsvor==2)
        {
			int count = 0;
				for(int j=0;j<nr_class-1;j++)
				{
					if (dec_values[j]>0)
						count  = j+1;
				}
			return model_->label[count];       	
        }
        else
        {
			int count = 1;
				for(int j=0;j<nr_class-1;j++)
				{
					if (dec_values[j]>0)
						count  += 1;
				}
			return model_->label[count-1];	    	
        }
	}
	else if(model_->param.solver_type == MCSVM_CS && (model_->param.npsvor>=3))
	{



			// if(model_->param.npsvor==6)  //CSOR all
			// {
			// 	int *count = new int[nr_class];
			// 	for(int k=0;k<nr_class;k++)
			// 	{
			// 		count[k] = 0;
			// 		for(int j=0;j<nr_class;j++)
			// 		{
			// 			if(j==k)
			// 				continue;
			// 			double yim = (k>j)?1:-1;
			// 			if (yim*(dec_values[k]+dec_values[j])>0)
			// 				count[k] +=1;
			// 		}
			// 	}
			// 	int dec_max_idx = 0;
			// 	for(int k=0;k<nr_class;k++)
			// 	{
			// 		if(count[k]>count[dec_max_idx])
			// 			dec_max_idx = k;
			// 	}			

			// 	return model_->label[dec_max_idx];	
			// } 
			// else 
			if(model_->param.npsvor>=3) //3 SSVOR SDM,  4 SSVOR  DCD,5  SNPSVOR DCD
			{
				int count = 1;
					for(int j=0;j<nr_class-1;j++)
					{
						// if (dec_values[j+1]+dec_values[j]>0)
						// 	{count  += 1;}
						if (dec_values[j+1]+dec_values[j]>0)
							count += 1;

					}
				return model_->label[count-1];	

			}
			else if(model_->param.npsvor==2)  //CSOR
			{
				int *count = new int[nr_class];
				for(int k=0;k<nr_class;k++)
				{
					count[k] = 0;
					for(int j=0;j<nr_class;j++)
					{
						if(j==k)
							continue;
						double yim = (k>j)?1:-1;
						if (yim*(dec_values[k]-dec_values[j])>0)
							count[k] +=1;
					}
				}
				int dec_max_idx = 0;
				for(int k=0;k<nr_class;k++)
				{
					if(count[k]>count[dec_max_idx])
						dec_max_idx = k;
				}			

				return model_->label[dec_max_idx];	
			}
			else  //Crammer-Singer Multi-class classification
			{
				int dec_max_idx = 0;
				for(i=1;i<nr_class;i++)
				{
					if(dec_values[i] > dec_values[dec_max_idx])
						dec_max_idx = i;
				}
				// info("prediction ");
				return model_->label[dec_max_idx];
			}
	}	
	else
	{
		int dec_max_idx = 0;
		for(i=1;i<nr_class;i++)
		{
			if(dec_values[i] > dec_values[dec_max_idx])
				dec_max_idx = i;
		}
		// info("prediction ");
		return model_->label[dec_max_idx];
	}
}

double predict(const model *model_, const feature_node *x)
{
	double *dec_values = Malloc(double, model_->nr_class);
	double label=predict_values(model_, x, dec_values);
	free(dec_values);
	return label;
}



double  predict_label(const feature_node *x, double *w,  double *label, int nr_w, int n, int nr_class)
{

	int idx;
	int i;
	double *dec_values = Malloc(double, nr_class);
	const feature_node *lx=x;
	for(i=0;i<nr_w;i++)
		dec_values[i] = 0;
	for(; (idx=lx->index)!=-1; lx++)
	{
		// the dimension of testing data may exceed that of training
			if(idx<=n)
				for(i=0;i<nr_w;i++)
					dec_values[i] += w[(idx-1)*nr_w+i]*lx->value;
	}

	int count = 1;
	for(int j=0;j<nr_class-1;j++)
	{
		// if (dec_values[j+1]+dec_values[j]>0)
		// 	{count  += 1;}
		if (dec_values[j+1]+dec_values[j]>0)
			count += 1;

	}
	return (double)label[count-1];	
}



double  predict_label_CrammerSinger(const feature_node *x, double *w,  double *label, int nr_w, int n, int nr_class)
{

	int idx;
	int i;
	double *dec_values = Malloc(double, nr_class);
	const feature_node *lx=x;
	for(i=0;i<nr_w;i++)
		dec_values[i] = 0;
	for(; (idx=lx->index)!=-1; lx++)
	{
		// the dimension of testing data may exceed that of training
			if(idx<=n)
				for(i=0;i<nr_w;i++)
					dec_values[i] += w[(idx-1)*nr_w+i]*lx->value;
	}

	int dec_max_idx = 0;
	for(i=1;i<nr_class;i++)
	{
		if(dec_values[i] > dec_values[dec_max_idx])
			dec_max_idx = i;
	}
	// info("prediction ");
	return (double)label[dec_max_idx];
}


double  predict_label_SVR(const feature_node *x, double *w,  double *label, int nr_w, int n, int nr_class)
{

	int idx;
	int i;
	double *dec_values = Malloc(double, nr_class);
	const feature_node *lx=x;
	for(i=0;i<nr_w;i++)
		dec_values[i] = 0;
	for(; (idx=lx->index)!=-1; lx++)
	{
		// the dimension of testing data may exceed that of training
			if(idx<=n)
				for(i=0;i<nr_w;i++)
					dec_values[i] += w[(idx-1)*nr_w+i]*lx->value;
	}

	int dec_min_idx = 0;
	double td = fabs(dec_values[0]-label[0]);
	for(i=1;i<nr_class;i++)
	{
		double ti = fabs(dec_values[0]-label[i]);
		if( ti < td)
			{
				dec_min_idx = i;
				td = ti;
			}
	}
	return (double)label[dec_min_idx];
}



double predict_probability(const struct model *model_, const struct feature_node *x, double* prob_estimates)
{
	if(check_probability_model(model_))
	{
		int i;
		int nr_class=model_->nr_class;
		int nr_w;
		if(nr_class==2)
			nr_w = 1;
		else
			nr_w = nr_class;

		double label=predict_values(model_, x, prob_estimates);
		for(i=0;i<nr_w;i++)
			prob_estimates[i]=1/(1+exp(-prob_estimates[i]));

		if(nr_class==2) // for binary classification
			prob_estimates[1]=1.-prob_estimates[0];
		else
		{
			double sum=0;
			for(i=0; i<nr_class; i++)
				sum+=prob_estimates[i];

			for(i=0; i<nr_class; i++)
				prob_estimates[i]=prob_estimates[i]/sum;
		}

		return label;
	}
	else
		return 0;
}

int save_model(const char *model_file_name, const struct model *model_)
{
	int i;
	int nr_feature=model_->nr_feature;
	int n;
	const parameter& param = model_->param;

	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	int w_size = n;
	FILE *fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	char *old_locale = setlocale(LC_ALL, NULL);
	if (old_locale)
	{
		old_locale = strdup(old_locale);
	}
	setlocale(LC_ALL, "C");

	int nr_w;
	if((model_->nr_class==2 && model_->param.solver_type != MCSVM_CS)
		||model_->param.solver_type ==L2R_SVOR||check_regression_model(model_))
		nr_w=1;
	else if(model_->param.solver_type ==L2R_SVMOP)
		nr_w = model_->nr_class -1;
	else
		nr_w=model_->nr_class;
    
	fprintf(fp, "solver_type %s\n", solver_type_table[param.solver_type]);
	fprintf(fp, "nr_class %d\n", model_->nr_class);
     
	if(model_->label)
	{
		fprintf(fp, "label");
		for(i=0; i<model_->nr_class; i++)
			fprintf(fp, " %d", model_->label[i]);
		fprintf(fp, "\n");
	}

	fprintf(fp, "nr_feature %d\n", nr_feature);

	fprintf(fp, "bias %.16g\n", model_->bias);

	fprintf(fp, "method %d\n", (int)param.npsvor);

	fprintf(fp, "w\n");
	for(i=0; i<w_size; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
			fprintf(fp, "%.16g ", model_->w[i*nr_w+j]);
		fprintf(fp, "\n");
	}

    if(model_->param.solver_type==L2R_SVOR)
	{
		fprintf(fp, "b\n");
		for(i=0; i<model_->nr_class-1; i++)
		{
			fprintf(fp, "%.16g ", model_->b[i]);
			fprintf(fp, "\n");
		}
	}
	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}

//
// FSCANF helps to handle fscanf failures.
// Its do-while block avoids the ambiguity when
// if (...)
//    FSCANF();
// is used
//
#define FSCANF(_stream, _format, _var)do\
{\
	if (fscanf(_stream, _format, _var) != 1)\
	{\
		fprintf(stderr, "ERROR: fscanf failed to read the model\n");\
		EXIT_LOAD_MODEL()\
	}\
}while(0)
// EXIT_LOAD_MODEL should NOT end with a semicolon.
#define EXIT_LOAD_MODEL()\
{\
	setlocale(LC_ALL, old_locale);\
	free(model_->label);\
	free(model_);\
	free(old_locale);\
	return NULL;\
}
struct model *load_model(const char *model_file_name)
{
	FILE *fp = fopen(model_file_name,"r");
	if(fp==NULL) return NULL;

	int i;
	int nr_feature;
	int n;
	int nr_class;
	int method;
	double bias;
	model *model_ = Malloc(model,1);
	parameter& param = model_->param;

	model_->label = NULL;

	char *old_locale = setlocale(LC_ALL, NULL);
	if (old_locale)
	{
		old_locale = strdup(old_locale);
	}
	setlocale(LC_ALL, "C");

	char cmd[81];
	while(1)
	{

		FSCANF(fp,"%80s",cmd);
		if(strcmp(cmd,"solver_type")==0)
		{
			FSCANF(fp,"%80s",cmd);
			int i;
			for(i=0;solver_type_table[i];i++)
			{
				if(strcmp(solver_type_table[i],cmd)==0)
				{
					param.solver_type=i;
					break;
				}
			}
			if(solver_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown solver type.\n");
				EXIT_LOAD_MODEL()
			}
		}
		else if(strcmp(cmd,"nr_class")==0)
		{
			FSCANF(fp,"%d",&nr_class);
			model_->nr_class=nr_class;
		}
		else if(strcmp(cmd,"nr_feature")==0)
		{
			FSCANF(fp,"%d",&nr_feature);
			model_->nr_feature=nr_feature;
		}
		else if(strcmp(cmd,"bias")==0)
		{
			FSCANF(fp,"%lf",&bias);
			model_->bias=bias;
		}
		else if(strcmp(cmd,"method")==0)
		{
			FSCANF(fp,"%d",&method);
			param.npsvor = method;
		}		
		else if(strcmp(cmd,"w")==0)
		{
			break;
		}
		else if(strcmp(cmd,"label")==0)
		{
			int nr_class = model_->nr_class;
			model_->label = Malloc(int,nr_class);
			for(int i=0;i<nr_class;i++)
				FSCANF(fp,"%d",&model_->label[i]);
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
			EXIT_LOAD_MODEL()
		}
	}

	nr_feature=model_->nr_feature;
	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	int w_size = n;
	int nr_w;
	// if(nr_class==2 && param.solver_type != MCSVM_CS)
	if((nr_class==2 && param.solver_type != MCSVM_CS)||param.solver_type ==L2R_SVOR||check_regression_model(model_))
		nr_w = 1;
	else if(param.solver_type ==L2R_SVMOP)
		nr_w = nr_class -1;
	else
		nr_w = nr_class;
	model_->w=Malloc(double, w_size*nr_w);
	for(i=0; i<w_size; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
			{
				FSCANF(fp, "%lf ", &model_->w[i*nr_w+j]);
				// info("%.3f ",model_->w[i*nr_w+j]);
			}
		if (fscanf(fp, "\n") !=0)
		{  
			fprintf(stderr, "ERROR: fscanf failed to read the model\n");
			EXIT_LOAD_MODEL()
		}
	}
    

	if(param.solver_type == L2R_SVOR)
	{   FSCANF(fp,"%80s",cmd);
		if(strcmp(cmd,"b")==0)
		{
		model_->b=Malloc(double, nr_class-1);
		for(i=0; i<nr_class-1; i++)
		{
			FSCANF(fp, "%lf ", &model_->b[i]);
			// info("%f\n",model_->b[i]);
		}
		}		
	}
	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0) return NULL;
	// info("%f\n",model_->b[0]);
	return model_;
}

int get_nr_feature(const model *model_)
{
	return model_->nr_feature;
}

int get_nr_class(const model *model_)
{
	return model_->nr_class;
}

void get_labels(const model *model_, int* label)
{
	if (model_->label != NULL)
		for(int i=0;i<model_->nr_class;i++)
			label[i] = model_->label[i];
}

// use inline here for better performance (around 20% faster than the non-inline one)
static inline double get_w_value(const struct model *model_, int idx, int label_idx) 
{
	int nr_class = model_->nr_class;
	int solver_type = model_->param.solver_type;
	const double *w = model_->w;

	if(idx < 0 || idx > model_->nr_feature)
		return 0;
	if(check_regression_model(model_))
		return w[idx];
	else 
	{
		if(label_idx < 0 || label_idx >= nr_class)
			return 0;
		if((nr_class==2 && solver_type != MCSVM_CS && solver_type != L2R_NPSVOR)||solver_type ==L2R_SVOR) //add
		// if(nr_class == 2 && solver_type != MCSVM_CS)
		{
			if(label_idx == 0)
				return w[idx];
			else
				return -w[idx];
		}
		else if(solver_type == L2R_SVMOP)
			return w[idx*(nr_class-1)+label_idx];
		else
			return w[idx*nr_class+label_idx];
	}
}

// feat_idx: starting from 1 to nr_feature
// label_idx: starting from 0 to nr_class-1 for classification models;
//            for regression models, label_idx is ignored.
double get_decfun_coef(const struct model *model_, int feat_idx, int label_idx)
{
	if(feat_idx > model_->nr_feature)
		return 0;
	return get_w_value(model_, feat_idx-1, label_idx);
}

double get_decfun_bias(const struct model *model_, int label_idx)
{
	int bias_idx = model_->nr_feature;
	double bias = model_->bias;
	if(bias <= 0)
		return 0;
	else
		return bias*get_w_value(model_, bias_idx, label_idx);
}

void free_model_content(struct model *model_ptr)
{
	if(model_ptr->w != NULL)
		free(model_ptr->w);
	if(model_ptr->label != NULL)
		free(model_ptr->label);
}

void free_and_destroy_model(struct model **model_ptr_ptr)
{
	struct model *model_ptr = *model_ptr_ptr;
	if(model_ptr != NULL)
	{
		free_model_content(model_ptr);
		free(model_ptr);
	}
}

void destroy_param(parameter* param)
{
	if(param->weight_label != NULL)
		free(param->weight_label);
	if(param->weight != NULL)
		free(param->weight);
	if(param->init_sol != NULL)
		free(param->init_sol);
}

const char *check_parameter(const problem *prob, const parameter *param)
{
	if(param->eps <= 0)
		return "eps <= 0";

	if(param->C <= 0)
		return "C <= 0";

	if(param->p < 0)
		return "p < 0";

	if(param->solver_type != L2R_LR
		&& param->solver_type != L2R_L2LOSS_SVC_DUAL
		&& param->solver_type != L2R_L2LOSS_SVC
		&& param->solver_type != L2R_L1LOSS_SVC_DUAL
		&& param->solver_type != MCSVM_CS
		&& param->solver_type != L1R_L2LOSS_SVC
		&& param->solver_type != L1R_LR
		&& param->solver_type != L2R_LR_DUAL
		&& param->solver_type != L2R_SVOR//my modification
		&& param->solver_type != L2R_NPSVOR
		&& param->solver_type != L2R_SVMOP
		&& param->solver_type != L2R_L2LOSS_SVR
		&& param->solver_type != L2R_L2LOSS_SVR_DUAL
		&& param->solver_type != L2R_L1LOSS_SVR_DUAL)
		{return "unknown solver type";}
		


	if(param->init_sol != NULL 
		&& param->solver_type != L2R_LR && param->solver_type != L2R_L2LOSS_SVC)
		return "Initial-solution specification supported only for solver L2R_LR and L2R_L2LOSS_SVC";

	return NULL;
}

int check_probability_model(const struct model *model_)
{
	return (model_->param.solver_type==L2R_LR ||
			model_->param.solver_type==L2R_LR_DUAL ||
			model_->param.solver_type==L1R_LR);
}

int check_regression_model(const struct model *model_)
{
	return (model_->param.solver_type==L2R_L2LOSS_SVR ||
			model_->param.solver_type==L2R_L1LOSS_SVR_DUAL ||
			model_->param.solver_type==L2R_L2LOSS_SVR_DUAL);
}

void set_print_string_function(void (*print_func)(const char*))
{
	if (print_func == NULL)
		liblinear_print_string = &print_string_stdout;
	else
		liblinear_print_string = print_func;
}

