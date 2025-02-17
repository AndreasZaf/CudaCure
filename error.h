#define MY_OK	1		/* avoid conflict with curses result code */
#define MY_ERR	0

/* error handling macros */
static int Erreturn(char *msg)
{
	printf("%s", msg);
	return MY_ERR;
}

#define IfErr(x) 	if ((x) == MY_ERR)
#define IfEOF(x)	if ((x) == EOF)

/*
 * macros handling D/C values
 */
#ifndef NO_DONTCARES
/*
 * we use IEEE NaN to represent don't care values -- ugly, but it works
 */
static long _nan = 0x7fffffff;
#define DC_VAL		((double)*(double *)(&_nan))
#define IS_DC(x)	isnan(x)

#ifdef sgi
#define isnan(x)	((x) == DC_VAL)
#endif

#ifdef WIN32
#define isnan(x)        ((x) == DC_VAL)
#endif

#endif /* !NO_DONTCARES*/ 