/*==================================================================================
 *    Copyright (C) 2024 Chengdu University of Technology.
 *    Copyright (C) 2024 Shaohuan Zu.
 *
 *    Filename：direct_cut.cpp
 *    Author：Shaohuan Zu
 *    Institute：Chengdu University of Technology
 *    Email：zushaohuan19@cdut.edu.cn
 *    Data：2024/08/20/
 *    Function：
 *
 *    This program is free software: you can redistribute it and/or modify it
 *    under the terms of the GNU General Public License as published by the Free
 *    Software Foundation, either version 3 of the License, or an later version.
 *=================================================================================*/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PI 3.1415926

#include "headobs.h"
#include "mpi.h"

int main(int argc, char *argv[]) {
    int myid, numprocs, namelen, index;
    MPI_Comm comm = MPI_COMM_WORLD;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(comm, &myid);
    MPI_Comm_size(comm, &numprocs);
    MPI_Get_processor_name(processor_name, &namelen);

    if (myid == 0) printf("Number of MPI thread is %d\n", numprocs);

    /*=========================================================
      Parameters of the time of the system...
      =========================================================*/
    time_t begin_time;
    //  time_t end_time;
    //  time_t last_time;

    clock_t start;
    clock_t end;

    //  float runtime=0.0;
    int nx, nz;
    int pml, Lc;
    float dx, dz;
    float rectime, dt;
    float f0;
    int Nf, freqintv;
    float freq0;
    int ns, r_n, itc;
    float sx0, shotdx, shotdep, moffsetx, s_distance, v_dir;
    float rx0, recdx, recdep, i_distance;
    int itn, iterb, ifreqb;
    int Ns;
    int ix, iz, is, it;
    FILE *fp, *fp1, *fp2;
    char v_file[80];
    char filename[100], filename1[100];
    int save_wf, inter_wf, simu;
    input_parameters(&nx, &nz, &pml, &Lc, &dx, &dz, &rectime, &dt, &f0, &ns, &sx0, &shotdx, &shotdep, &r_n, &rx0, &recdx, &recdep, &simu, &s_distance, &save_wf, &inter_wf, v_file);

    int itmax = (int)(rectime / dt);
    if (itmax % 10 != 0) itmax = itmax + 1;

    printf("irectime = %f, dt= %f, itmax = %d\n", rectime, dt, itmax);
    float *vp;
    float *seisdata;
    vp = (float *)malloc(nx * nz * sizeof(float));
    seisdata = (float *)malloc(r_n * itmax * sizeof(float));
    // memset(seisdata,0,sizeof(float)*itmax*nx);
    // for (it=0;it<itmax*nx;it++)
    // seisdata[it]=0;

    printf("%s\n", v_file);
    if (strcmp(v_file, "none") == 0) {
        sprintf(v_file, argv[1]);
    }
    printf("%s\n", v_file);

    if ((fp = fopen(v_file, "rb")) == NULL)
        printf("Cannot obtain the velocity file\n");
    else {
        fread(vp, sizeof(float), nx * nz, fp);
    }
    fclose(fp);

    v_dir = vp[(int)(shotdep / dz)];
    // fp1 = fopen("time.txt","w");
    for (is = 0; is < ns; is++) {
        if (simu == 1)
            sprintf(filename, "./forward_output/%dsource_seismogram_rms_simu.bin", is + 1);
        else
            sprintf(filename, "./forward_output/%dsource_seismogram_rms.bin", is + 1);
        fp = fopen(filename, "rb");
        fread(&seisdata[0], sizeof(float), itmax * r_n, fp);
        fclose(fp);
        for (ix = 0; ix < r_n; ix++) {
            itc = (int)abs(((ix * recdx - (sx0 + is * shotdx)) / v_dir / dt));
            if (itc < itmax)
                for (it = 0; it < itc; it++)
                    seisdata[it + ix * itmax] = 0.0;
            else
                for (it = 0; it < itmax; it++)
                    seisdata[it + ix * itmax] = 0.0;
        };
        if (simu == 1)
            sprintf(filename1, "./forward_output/%dsource_seismogram_rms_simu_cut.bin", is + 1);
        else
            sprintf(filename1, "./forward_output/%dsource_seismogram_rms_cut.bin", is + 1);
        printf("%s\n", filename1);
        fp2 = fopen(filename1, "wb");
        fwrite(&seisdata[0], sizeof(float), itmax * r_n, fp2);
        fclose(fp2);
    }
    // fclose(fp1);
}

void input_parameters(int *nx, int *nz, int *pml, int *Lc, float *dx, float *dz, float *rectime, float *dt, float *f0, int *ns, float *sx0, float *shotdx, float *shotdep, int *r_n, float *rx0, float *recdx, float *recdep, int *simu, float *s_distance, int *save_wf, int *inter_wf, char *v_file) {
    char strtmp[256];
    FILE *fp = fopen("./parameter.txt", "r");
    if (fp == 0) {
        printf("Cannot open the parameters1 file!\n");
        exit(0);
    }

    fgets(strtmp, 256, fp);
    fscanf(fp, "\n");
    fscanf(fp, "%d", nx);
    fscanf(fp, "\n");

    fgets(strtmp, 256, fp);
    fscanf(fp, "\n");
    fscanf(fp, "%d", nz);
    fscanf(fp, "\n");

    fgets(strtmp, 256, fp);
    fscanf(fp, "\n");
    fscanf(fp, "%d", pml);
    fscanf(fp, "\n");

    fgets(strtmp, 256, fp);
    fscanf(fp, "\n");
    fscanf(fp, "%d", Lc);
    fscanf(fp, "\n");

    fgets(strtmp, 256, fp);
    fscanf(fp, "\n");
    fscanf(fp, "%f", dx);
    fscanf(fp, "\n");

    fgets(strtmp, 256, fp);
    fscanf(fp, "\n");
    fscanf(fp, "%f", dz);
    fscanf(fp, "\n");

    fgets(strtmp, 256, fp);
    fscanf(fp, "\n");
    fscanf(fp, "%f", rectime);
    fscanf(fp, "\n");

    fgets(strtmp, 256, fp);
    fscanf(fp, "\n");
    fscanf(fp, "%f", dt);
    fscanf(fp, "\n");

    fgets(strtmp, 256, fp);
    fscanf(fp, "\n");
    fscanf(fp, "%f", f0);
    fscanf(fp, "\n");

    fgets(strtmp, 256, fp);
    fscanf(fp, "\n");
    fscanf(fp, "%d", ns);
    fscanf(fp, "\n");

    fgets(strtmp, 256, fp);
    fscanf(fp, "\n");
    fscanf(fp, "%f", sx0);
    fscanf(fp, "\n");

    fgets(strtmp, 256, fp);
    fscanf(fp, "\n");
    fscanf(fp, "%f", shotdx);
    fscanf(fp, "\n");

    fgets(strtmp, 256, fp);
    fscanf(fp, "\n");
    fscanf(fp, "%f", shotdep);
    fscanf(fp, "\n");

    fgets(strtmp, 256, fp);
    fscanf(fp, "\n");
    fscanf(fp, "%d", r_n);
    fscanf(fp, "\n");

    fgets(strtmp, 256, fp);
    fscanf(fp, "\n");
    fscanf(fp, "%f", rx0);
    fscanf(fp, "\n");

    fgets(strtmp, 256, fp);
    fscanf(fp, "\n");
    fscanf(fp, "%f", recdx);
    fscanf(fp, "\n");

    fgets(strtmp, 256, fp);
    fscanf(fp, "\n");
    fscanf(fp, "%f", recdep);
    fscanf(fp, "\n");

    fgets(strtmp, 256, fp);
    fscanf(fp, "\n");
    fscanf(fp, "%d", simu);
    fscanf(fp, "\n");

    fgets(strtmp, 256, fp);
    fscanf(fp, "\n");
    fscanf(fp, "%f", s_distance);
    fscanf(fp, "\n");

    fgets(strtmp, 256, fp);
    fscanf(fp, "\n");
    fscanf(fp, "%d", save_wf);
    fscanf(fp, "\n");

    fgets(strtmp, 256, fp);
    fscanf(fp, "\n");
    fscanf(fp, "%d", inter_wf);
    fscanf(fp, "\n");

    fgets(strtmp, 256, fp);
    fscanf(fp, "\n");
    fscanf(fp, "%s", v_file);
    fscanf(fp, "\n");

    return;
}

/*==========================================================
  This subroutine is used for initializing the true model...
  ===========================================================*/

void get_acc_model(float *vp, float *rho, int ntp, int ntx, int ntz, int pml, char *v0_file) {
    int ip, ipp, iz, ix;
    // THE MODEL
    FILE *fp;

    char vv_file[90];
    sprintf(vv_file, "%s", v0_file);
    // printf("%s\n",vv_file);

    if ((fp = fopen(vv_file, "rb")) == NULL)
        printf("Cannot read the velocity file --- %s\n", vv_file);
    else {
        for (ix = pml; ix < ntx - pml; ix++) {
            for (iz = pml; iz < ntz - pml; iz++) {
                ip = iz * ntx + ix;
                fread(&vp[ip], sizeof(float), 1, fp);
            }
        }
    }
    fclose(fp);

    for (iz = 0; iz <= pml - 1; iz++) {
        for (ix = 0; ix <= pml - 1; ix++) {
            ip = iz * ntx + ix;
            ipp = pml * ntx + pml;

            vp[ip] = vp[ipp];
        }

        for (ix = pml; ix <= ntx - pml - 1; ix++) {
            ip = iz * ntx + ix;
            ipp = pml * ntx + ix;

            vp[ip] = vp[ipp];
        }

        for (ix = ntx - pml; ix < ntx; ix++) {
            ip = iz * ntx + ix;
            ipp = pml * ntx + ntx - pml - 1;

            vp[ip] = vp[ipp];
        }
    }

    for (iz = pml; iz <= ntz - pml - 1; iz++) {
        for (ix = 0; ix <= pml - 1; ix++) {
            ip = iz * ntx + ix;
            ipp = iz * ntx + pml;

            vp[ip] = vp[ipp];
        }

        for (ix = ntx - pml; ix < ntx; ix++) {
            ip = iz * ntx + ix;
            ipp = iz * ntx + ntx - pml - 1;

            vp[ip] = vp[ipp];
        }
    }

    for (iz = ntz - pml; iz < ntz; iz++) {
        for (ix = 0; ix <= pml - 1; ix++) {
            ip = iz * ntx + ix;
            ipp = (ntz - pml - 1) * ntx + pml;

            vp[ip] = vp[ipp];
        }

        for (ix = pml; ix <= ntx - pml - 1; ix++) {
            ip = iz * ntx + ix;
            ipp = (ntz - pml - 1) * ntx + ix;

            vp[ip] = vp[ipp];
        }

        for (ix = ntx - pml; ix < ntx; ix++) {
            ip = iz * ntx + ix;
            ipp = (ntz - pml - 1) * ntx + ntx - pml - 1;

            vp[ip] = vp[ipp];
        }
    }
    ///////////
    /*	fp=fopen("../input/acc_rho.bin","rb");
            for(ix=pml;ix<ntx-pml;ix++)
            {
                    for(iz=pml;iz<ntz-pml;iz++)
                    {
                            ip=iz*ntx+ix;
                            fread(&rho[ip],sizeof(float),1,fp);

                            rho[ip]=rho[ip];
                    }
            }
            fclose(fp);

            for(iz=0;iz<=pml-1;iz++)
            {

                    for(ix=0;ix<=pml-1;ix++)
                    {
                            ip=iz*ntx+ix;
                            ipp=pml*ntx+pml;

                            rho[ip]=rho[ipp];
                    }

                    for(ix=pml;ix<=ntx-pml-1;ix++)
                    {
                            ip=iz*ntx+ix;
                            ipp=pml*ntx+ix;

                            rho[ip]=rho[ipp];
                    }

                    for(ix=ntx-pml;ix<ntx;ix++)
                    {
                            ip=iz*ntx+ix;
                            ipp=pml*ntx+ntx-pml-1;

                            rho[ip]=rho[ipp];
                    }
            }

            for(iz=pml;iz<=ntz-pml-1;iz++)
            {
                    for(ix=0;ix<=pml-1;ix++)
                    {
                            ip=iz*ntx+ix;
                            ipp=iz*ntx+pml;

                            rho[ip]=rho[ipp];
                    }

                    for(ix=ntx-pml;ix<ntx;ix++)
                    {
                            ip=iz*ntx+ix;
                            ipp=iz*ntx+ntx-pml-1;

                            rho[ip]=rho[ipp];
                    }

            }

            for(iz=ntz-pml;iz<ntz;iz++)
            {

                    for(ix=0;ix<=pml-1;ix++)
                    {
                            ip=iz*ntx+ix;
                            ipp=(ntz-pml-1)*ntx+pml;

                            rho[ip]=rho[ipp];
                    }

                    for(ix=pml;ix<=ntx-pml-1;ix++)
                    {
                            ip=iz*ntx+ix;
                            ipp=(ntz-pml-1)*ntx+ix;

                            rho[ip]=rho[ipp];
                    }

                    for(ix=ntx-pml;ix<ntx;ix++)
                    {
                            ip=iz*ntx+ix;
                            ipp=(ntz-pml-1)*ntx+ntx-pml-1;

                            rho[ip]=rho[ipp];
                    }
            }
    */
    for (ip = 0; ip < ntp; ip++) {
        rho[ip] = 1000.0;
    }

    return;
}

