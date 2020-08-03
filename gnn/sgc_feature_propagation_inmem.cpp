#include<string>

#include "graphchi_basic_includes.hpp"

#include "../example_apps/matrix_factorization/matrixmarket/mmio.h"
#include "../example_apps/matrix_factorization/matrixmarket/mmio.c"

#include "Eigen/Dense"

#include "Eigen/Sparse"
#include "unsupported/Eigen/SparseExtra"


using namespace Eigen;

typedef VectorXd vector;


using namespace graphchi;


int dim = 1433;


struct FeatureVector {
	vector val;
	int deg;

	FeatureVector() {
		deg = 0;
		val = vector::Zero(dim);
	}

	void set(int index, float value) {
		val[index] = value;
	}

	float get(int index) {
		return val[index];
	}
};


template<typename FeatureVector> struct  MMOutputter {
	MMOutputter(std::string fname, uint nrow, uint ncol, std::vector<FeatureVector> & feature_vector_collection)  {
		bool sparse=true;
		
		MM_typecode matcode;
		
		mm_initialize_typecode(&matcode);
		mm_set_matrix(&matcode);
		
		if (sparse)
    			mm_set_coordinate(&matcode);
  		else
    			mm_set_array(&matcode);
  		
		mm_set_real(&matcode);

    		FILE * outf = fopen(fname.c_str(), "w");
    		mm_write_banner(outf, matcode);

    		if (sparse)
      			mm_write_mtx_crd_size(outf, nrow, ncol, nrow * ncol);
    		else
      			mm_write_mtx_array_size(outf, nrow, ncol);

    		for (uint i=0; i < nrow; i++) {
			for(int j=0; j < ncol; j++) {
				if (sparse)
					fprintf(outf, "%d %d %12.8g\n", i, j, feature_vector_collection[i].get(j));
        			else {
          				fprintf(outf, "%1.12e ", feature_vector_collection[i].get(j));
          
					if (j == ncol - 1)
	    					fprintf(outf, "\n");
        			}
      			}
      		}
		fclose(outf);
	}
};


typedef FeatureVector VertexDataType;
typedef float EdgeDataType;

std::vector<FeatureVector> feature_vector_collection;

FILE * f;
int ret_code;
MM_typecode matcode;


struct SGCFeaturePropagationInMemProgram : public GraphChiProgram<VertexDataType, EdgeDataType> { 
	
	void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &context) {
		FeatureVector & self_feature = feature_vector_collection[vertex.id()];
    
		if (context.iteration == 0) {
			feature_vector_collection[vertex.id()].deg = vertex.num_edges();
		}

    		if (vertex.num_edges() == 0)
      			return;

    		vector self_feature_nxt = vector::Zero(dim);

    		for(int e = 0; e < vertex.num_edges(); e++) {
	    		float weight = vertex.edge(e)->get_data() + 1; // FIXME                
	    		assert(weight == 1);
	    
	   	 	FeatureVector & nbr_feature = feature_vector_collection[vertex.edge(e)->vertex_id()];
	    		self_feature_nxt += (weight / std::sqrt((1 + vertex.num_edges()) * (1 + nbr_feature.deg))) * nbr_feature.val;
    		}

    		self_feature.val = (1 / self_feature.deg) * self_feature.val + self_feature_nxt;
	}
};


int main(int argc, const char **argv) {
	graphchi_init(argc, argv);

	metrics m("SGC_feature_propagation");

	std::string feature_file = get_option_string("feature-file");

	std::cout << feature_file << std::endl;
	
	if ((f = fopen(feature_file.c_str(), "r")) == NULL) {
		logstream(LOG_ERROR) << "Could not open file: " << feature_file << ", error: " << strerror(errno) << std::endl;
		exit(1);
    	}

	if (mm_read_banner(f, &matcode) != 0) {
		logstream(LOG_ERROR) << "Could not process Matrix Market banner. File: " << feature_file << std::endl;
		logstream(LOG_ERROR) << "Matrix must be in the Matrix Market format. " << std::endl;
		exit(1);
	}

	if (mm_is_complex(matcode))
    	{
        	logstream(LOG_ERROR) << "Sorry, this application does not support complex values." << std::endl;
        	logstream(LOG_ERROR) << "Market Market type: " << mm_typecode_to_str(matcode) << std::endl;
        	exit(1);
	}

	uint nrow, ncol;
	size_t num_nz;
	
	if (mm_is_sparse(matcode)) {
		if ((ret_code = mm_read_mtx_crd_size(f, &nrow, &ncol, &num_nz)) !=0) {
        		logstream(LOG_ERROR) << "Failed reading matrix size: error=" << ret_code << std::endl;
        		exit(1);
    		}
	} else {
		if ((ret_code = mm_read_mtx_array_size(f, &nrow, &ncol)) != 0) {
			logstream(LOG_ERROR) << "Failed reading matrix size: error=" << ret_code << std::endl;
			exit(1);
		}
    		num_nz = nrow * ncol;
	}	

	logstream(LOG_INFO) << "Starting to read matrix-market input. Matrix dimensions: " << nrow << " x " << ncol << ", non-zeros: " << num_nz << std::endl;

	feature_vector_collection.resize(nrow);

	if (dim != (int) ncol)
    logstream(LOG_FATAL) << "Wrong matrix size detected, command line argument should be --dim=" << ncol << ", instead of " << dim << std::endl;

	uint row, col;
	double val;
	
	for (int i=0; i < (int) num_nz; i++) {
    		if (mm_is_sparse(matcode)) {
      			ret_code = fscanf(f, "%u %u %lg\n", &row, &col, &val);
      		
			if (ret_code != 3)
        			logstream(LOG_FATAL) << "Error reading input line " << i << std::endl;
      		
			row--; 
			col--;

     	 		assert(row >= 0 && row < nrow);
      			assert(col >= 0 && col < ncol);
      		
			feature_vector_collection[row].set(col, val);
    		}
    		else {
      			ret_code = fscanf(f, "%lg", &val);
      			
			if (ret_code != 1)
        			logstream(LOG_FATAL) << "Error reading nnz " << i << std::endl;
      			
			row = i / ncol;
      			col = i % ncol;
      			
			feature_vector_collection[row].set(col, val);
    		}
  	}

  	logstream(LOG_INFO) << "Feature vectors from file: loaded matrix of size " << nrow << " x " << ncol << " from file: " << feature_file << " total of " << num_nz << " entries. " << std::endl;
 	
	fclose(f);

	std::string edge_file = get_option_string("edge-file");
	
	bool preexisting_shard;
    	int nshard = convert_if_notexists<vid_t>(edge_file, get_option_string("nshard", "auto"), preexisting_shard);
	
	int niter = get_option_int("niter", 4);
	bool scheduler = false;
	SGCFeaturePropagationInMemProgram program;
	graphchi_engine<VertexDataType, EdgeDataType> engine(edge_file, nshard, scheduler, m);
	engine.set_modifies_inedges(false);
	engine.set_modifies_outedges(false);
	engine.set_enable_deterministic_parallelism(false);

	engine.run(program, niter);

	MMOutputter<FeatureVector>  mmoutput(feature_file + "-smooth.mtx", nrow, ncol, feature_vector_collection); // FIXME

	metrics_report(m);

	SparseMatrix<float> feature_matrix(nrow, ncol);
	if (!loadMarket(feature_matrix, feature_file)) {
		logstream(LOG_ERROR) << "Eigen could not load sparse matrix from " << feature_file << std::endl;
	}

	logstream(LOG_INFO) << "Eigen read a " << feature_matrix.rows() << " x " << feature_matrix.cols() << " sparse matrix with " << feature_matrix.nonZeros() << " non-zero entries from " << feature_file << std::endl;

	return 0;	
}

