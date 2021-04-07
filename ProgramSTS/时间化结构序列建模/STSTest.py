import tensorflow_probability as tfp
import pandas as pd
import numpy as np
import tensorflow as tf

def plot_forecast(x, y,
                  forecast_mean, forecast_scale, forecast_samples,
                  title, x_locator=None, x_formatter=None):
  import seaborn as sns
  from matplotlib import pylab as plt
  import numpy as np
  """Plot a forecast distribution against the 'true' time series."""
  colors = sns.color_palette()
  c1, c2 = colors[0], colors[1]
  fig = plt.figure(figsize=(12, 6))
  ax = fig.add_subplot(1, 1, 1)

  num_steps = len(y)
  num_steps_forecast = forecast_mean.shape[-1]
  num_steps_train = num_steps - num_steps_forecast


  ax.plot(x, y, lw=2, color=c1, label='ground truth')

  forecast_steps = np.arange(
      x[num_steps_train],
      x[num_steps_train]+num_steps_forecast,
      dtype=x.dtype)

  ax.plot(forecast_steps, forecast_samples.T, lw=1, color=c2, alpha=0.1)

  ax.plot(forecast_steps, forecast_mean, lw=2, ls='--', color=c2,
           label='forecast')
  ax.fill_between(forecast_steps,
                   forecast_mean-2*forecast_scale,
                   forecast_mean+2*forecast_scale, color=c2, alpha=0.2)

  ymin, ymax = min(np.min(forecast_samples), np.min(y)), max(np.max(forecast_samples), np.max(y))
  yrange = ymax-ymin
  ax.set_ylim([ymin - yrange*0.1, ymax + yrange*0.1])
  ax.set_title("{}".format(title))
  ax.legend()

  if x_locator is not None:
    ax.xaxis.set_major_locator(x_locator)
    ax.xaxis.set_major_formatter(x_formatter)
    fig.autofmt_xdate()

  return fig, ax

mainTrainData = np.array(pd.read_excel('maintrain.xlsx'))
convTrainData = np.array(pd.read_excel('convtrain.xlsx'))
convTestData = np.array(pd.read_excel('convtest.xlsx'))
convData = np.concatenate((convTrainData, convTestData), axis=0)
# #steps=int(self.lineEdit.text())
num_variational_steps = 20
num_variational_steps = int(num_variational_steps)

mainTrainData = np.reshape(mainTrainData,(mainTrainData.shape[0],))
convTrainData = np.reshape(convTrainData,(convTrainData.shape[0],))
convTestData = np.reshape(convTestData,(convTestData.shape[0],))
convData=np.reshape(convData,(convData.shape[0],))

mainTrainData=pd.Series(mainTrainData)
convTrainData=pd.Series(convTrainData)
convTestData=pd.Series(convTestData)
#
seasonal=[]
seasonalStep=[]
seasonalModuleName=[]
row = 1
for i in range(row):
    seasonalModuleName.append(str('tableWidget_seasonalModule_STS.item(i, 0).text()'))
    seasonalStep.append(int(24#tableWidget_seasonalModule_STS.item(i, 1).text()
                            ))
    #self.seasonalStepSon.append(int(self.tableWidget.item(i, 1).text()))
convNumber=1
conv_effct=[]

for i in range(convNumber):
    b=tfp.sts.LinearRegression(
        design_matrix=tf.reshape(convData-np.mean(convData),(-1,1),name='s2'
    ))
    conv_effct.append(b)
seasonal_effect=[]
for i in range(row):
    a=tfp.sts.Seasonal(num_seasons=24,observed_time_series=mainTrainData,name='seasonalModuleName[i]')
    seasonal_effect.append(a)
#AR = tfp.sts.Autoregressive(order=1, observed_time_series=mainTrainData, name='autoregressive')
modelList=[]
for i in conv_effct:
    modelList.append(i)
for i in seasonal_effect:
    modelList.append(i)
# if checkBox_ARModuleIf_STS.isChecked():
#     modelList.append(AR)
model = tfp.sts.Sum(modelList,observed_time_series=mainTrainData)
print(modelList)
variational_posteriors = tfp.sts.build_factored_surrogate_posterior(model=model)

##搭建时间序列模型
# hour_of_day_effect = tfp.sts.Seasonal(
#         num_seasons=24,
#         observed_time_series=mainTrainData,
#         name='hour_of_day_effect')
# temperature_effect = tfp.sts.LinearRegression(
#       design_matrix=tf.reshape(convData - np.mean(convData),
#                                (-1, 1)), name='temperature_effect')
# autoregressive = tfp.sts.Autoregressive(
#       order=1,
#       observed_time_series=mainTrainData,
#       name='autoregressive')
# list1=[]
#
#
#
# list1.append(hour_of_day_effect)
# list1.append(temperature_effect)





#model = tfp.sts.Sum([hour_of_day_effect,temperature_effect,autoregressive],observed_time_series=pavementTemperature_train)
model=tfp.sts.Sum(modelList,observed_time_series=mainTrainData)
variational_posteriors = tfp.sts.build_factored_surrogate_posterior(model=model)

optimizer = tf.optimizers.Adam(learning_rate=0.1)

elbo_loss_curve = tfp.vi.fit_surrogate_posterior(
    target_log_prob_fn=model.joint_log_prob(observed_time_series=mainTrainData),
    surrogate_posterior=variational_posteriors,
    optimizer=optimizer,
    num_steps=num_variational_steps)
elbo_loss_curve = np.array(elbo_loss_curve)
q_samples_demand_ = variational_posteriors.sample(50)
# 进行预测
demand_forecast_dist = tfp.sts.forecast(model=model, observed_time_series=mainTrainData,
                                        parameter_samples=q_samples_demand_, num_steps_forecast=len(convTestData))
num_samples = 30

(
    demand_forecast_mean,
    demand_forecast_scale,
    demand_forecast_samples
) = (
    demand_forecast_dist.mean().numpy()[..., 0],
    demand_forecast_dist.stddev().numpy()[..., 0],
    demand_forecast_dist.sample(num_samples).numpy()[..., 0]
)
date = np.arange('2013-12-26T15:00', '2014-01-07T03:00', dtype='datetime64[h]')
fig, ax = plot_forecast(date,convData,
                        demand_forecast_mean,
                        demand_forecast_scale,
                        demand_forecast_samples,
                        title="Pavement Temperature forecast")

fig.savefig('testblueline.jpg')

