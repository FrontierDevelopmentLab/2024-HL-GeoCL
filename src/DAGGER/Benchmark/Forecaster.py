import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator as Interpolator

"""
 Generate forecast

A function which takes in the model, dataloader and generates forecast.

Will also Generate Weimer forecast at Supermag locations

"""


def Forecaster(dataloader,models,dbe_mean,dbe_std,dbn_mean,dbn_std,target_index):
    db_means = {'dbe': dbe_mean, 'dbn': dbn_mean}
    db_stds = {'dbe': dbe_std, 'dbn': dbn_std}
    with torch.no_grad():
        Predictions = {'dbe':[[] for _ in models],'dbn':[[] for _ in models]} 
        Targets = {'dbe':[],'dbn':[]} 
        dates = {'dbe':[],'dbn':[]} 
        Coefficients = {'dbe':[[] for _ in models],'dbn':[[] for _ in models]} 
        MLT_sup = {'dbe':[],'dbn':[]} 
        Mcolat_sup = {'dbe':[],'dbn':[]} 
        for k in ["dbe", "dbn"]:
            for (
                past_omni,
                past_supermag,
                future_supermag,
                past_dates,
                future_dates,
                (mlt, mcolat),
            ) in dataloader:

                past_omni = past_omni.cuda()
                past_supermag = past_supermag.cuda()
                future_supermag = future_supermag.cuda()
                past_dates = past_dates.cuda()
                future_dates = future_dates.cuda()
                mlt = mlt.cuda()
                mcolat = mcolat.cuda()
                
                db_target = future_supermag[:,:,:,target_index[k]]
                Targets[k].append(db_target.detach().cpu().numpy())
                MLT_sup[k].append(mlt.cpu().numpy())
                Mcolat_sup[k].append(mcolat.cpu().numpy())
                dates[k].append(future_dates.cpu().numpy()) 

                for (i, model) in enumerate(models):
                    _, _coeffsdb, db_pred = model(past_omni, past_supermag, mlt, mcolat, past_dates, future_dates)

                    m = {'dbe': 0, 'dbn': 1}
                    _coeffsdb = _coeffsdb[..., m[k]]
                    db_pred = db_pred[..., m[k]]

                    Coefficients[k][i].append(_coeffsdb.detach().cpu().numpy())
                    Predictions[k][i].append(db_pred.detach().cpu().numpy())
                    
    All_times_coeff = {'dbe':[[] for _ in models],'dbn':[[] for _ in models]} 
    Date_arr = {'dbe':[],'dbn':[]} 
    MLT_sup_all = {'dbe':[],'dbn':[]} 
    Mcolat_sup_all = {'dbe':[],'dbn':[]} 
    variance = {}
    
    for k in Predictions.keys():
        for i, model in enumerate(models):
            Predictions[k][i] = np.concatenate(Predictions[k][i],axis=0)
            All_times_coeff[k][i] = np.concatenate(Coefficients[k][i],axis=0)
            Predictions[k][i] = Predictions[k][i]*db_stds[k] + db_means[k]

        Targets[k] = np.squeeze(np.concatenate(Targets[k],axis=0),axis=1)
        Targets[k] = Targets[k]*db_stds[k] + db_means[k]
        Date_arr[k] = np.concatenate(dates[k],axis=0)
        MLT_sup_all[k] = np.concatenate(MLT_sup[k],axis=0).squeeze(1)
        Mcolat_sup_all[k] = np.concatenate(Mcolat_sup[k],axis=0).squeeze(1)
        variance[k] = np.var(Predictions[k], axis = 0)
        Predictions[k] = np.mean(Predictions[k], axis = 0)
        All_times_coeff[k] = np.mean(All_times_coeff[k], axis = 0)

    return Predictions,Targets,All_times_coeff,Date_arr,MLT_sup_all,Mcolat_sup_all, variance


class Custom_interpolator:
    def __init__(self, inx, iny, indata):
        self.inx = inx[::-1]
        self.iny = iny[::-1]
        self.indata = indata[::-1, ::-1]
        self.fun = Interpolator(
            (self.inx, self.iny), self.indata.T, bounds_error=False, fill_value=np.nan
        )

    def __call__(self, points):
        return self.fun(points)


def Generate_weimer_forecast(weimermlt, weimercolat, weimerongrid, targmlt, targmcolat):
    weimer_on_grid = Custom_interpolator(weimermlt, weimercolat, weimerongrid)
    sample_points = np.asarray(list(zip(targmlt, targmcolat)))
    return weimer_on_grid(sample_points)


def Generate_complete_weimer_forecast(
    weimermlt, weimercolat, weimerongrid, mltarray, colatarray
):
    weimer_preds_all_time = []
    for i in np.arange(len(weimerongrid)):
        weimer_preds_all_time.append(
            Generate_weimer_forecast(
                weimermlt, weimercolat, weimerongrid[i], mltarray[i], colatarray[i]
            )
        )
    return np.asarray(weimer_preds_all_time)
