import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import sts

###画图函数
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

####画组成的函数
def plot_components(dates,
                    component_means_dict,
                    component_stddevs_dict,
                    x_locator=None,
                    x_formatter=None):
  import seaborn as sns
  from matplotlib import pylab as plt
  import numpy as np
  import collections
  """Plot the contributions of posterior components in a single figure."""
  colors = sns.color_palette()
  c1, c2 = colors[0], colors[1]

  axes_dict = collections.OrderedDict()
  num_components = len(component_means_dict)
  fig = plt.figure(figsize=(12, 2.5 * num_components))
  for i, component_name in enumerate(component_means_dict.keys()):
    component_mean = component_means_dict[component_name]
    component_stddev = component_stddevs_dict[component_name]

    ax = fig.add_subplot(num_components,1,1+i)
    ax.plot(dates, component_mean, lw=2)
    ax.fill_between(dates,
                     component_mean-2*component_stddev,
                     component_mean+2*component_stddev,
                     color=c2, alpha=0.5)
    ax.set_title(component_name)
    if x_locator is not None:
      ax.xaxis.set_major_locator(x_locator)
      ax.xaxis.set_major_formatter(x_formatter)
    axes_dict[component_name] = ax
  fig.autofmt_xdate()
  fig.tight_layout()
  return fig, axes_dict

##预测画图函数
def plot_one_step_predictive(dates, observed_time_series,
                             one_step_mean, one_step_scale,
                             x_locator=None, x_formatter=None):
  import seaborn as sns
  from matplotlib import pylab as plt
  import numpy as np
  """Plot a time series against a model's one-step predictions."""

  colors = sns.color_palette()
  c1, c2 = colors[0], colors[1]

  fig=plt.figure(figsize=(12, 6))
  ax = fig.add_subplot(1,1,1)
  num_timesteps = one_step_mean.shape[-1]
  ax.plot(dates, observed_time_series, label="observed time series", color=c1)
  ax.plot(dates, one_step_mean, label="one-step prediction", color=c2)
  ax.fill_between(dates,
                  one_step_mean - one_step_scale,
                  one_step_mean + one_step_scale,
                  alpha=0.1, color=c2)
  ax.legend()

  if x_locator is not None:
    ax.xaxis.set_major_locator(x_locator)
    ax.xaxis.set_major_formatter(x_formatter)
    fig.autofmt_xdate()
  fig.tight_layout()
  return fig, ax

def plot_compare_variables(dates, y1, y2, label1, label2, suptitle,num_forecast_steps):
  import seaborn as sns
  from matplotlib import pylab as plt
  import numpy as np
  colors = sns.color_palette()
  c1, c2 = colors[0], colors[1]
  fig = plt.figure(figsize=(12, 6))
  ax = fig.add_subplot(2, 1, 1)
  ax.plot(dates[:-num_forecast_steps],
            y1[:-num_forecast_steps], lw=2, label=label1)
  ax.set_ylabel(label1)

  ax = fig.add_subplot(2, 1, 2)
  ax.plot(dates[:-num_forecast_steps],
            y2[:-num_forecast_steps], lw=2, label=label2, c=c2)
  ax.set_ylabel(label2)
  fig.suptitle(suptitle,
                 fontsize=15)
  fig.autofmt_xdate()

###程序运行
##读取数据
demand_loc = mdates.WeekdayLocator(byweekday=mdates.WE)
demand_fmt = mdates.DateFormatter('%a %b %d')

trainData=pd.read_excel('训练集.xlsx')
testData=pd.read_excel('测试集.xlsx')
##设置时间
date = np.arange('2013-12-26T15:00', '2014-01-07T03:00', dtype='datetime64[h]')
pavementTemperature_train=trainData['路表温度']
airTemperature_train=trainData['大气温度']
pavementTemperature_test=testData['路表温度']
airTemperature_test=testData['大气温度']
airTemperature=np.concatenate((airTemperature_train,airTemperature_test),axis = 0)
pavementTemperature=np.concatenate((pavementTemperature_train,pavementTemperature_test),axis=0)

##搭建时间序列模型
hour_of_day_effect = sts.Seasonal(
        num_seasons=24,
        observed_time_series=pavementTemperature_train,
        name='hour_of_day_effect')
temperature_effect = sts.LinearRegression(
      design_matrix=tf.reshape(airTemperature - np.mean(airTemperature),
                               (-1, 1)), name='temperature_effect')
autoregressive = sts.Autoregressive(
      order=1,
      observed_time_series=pavementTemperature_train,
      name='autoregressive')
list1=[]
list1.append(hour_of_day_effect)
list1.append(temperature_effect)
model = sts.Sum([hour_of_day_effect,temperature_effect,autoregressive],observed_time_series=pavementTemperature_train)
model=sts.Sum(list1,observed_time_series=pavementTemperature_train)
variational_posteriors = tfp.sts.build_factored_surrogate_posterior(model=model)


num_variational_steps = 400
num_variational_steps = int(num_variational_steps)

optimizer = tf.optimizers.Adam(learning_rate=0.1)

elbo_loss_curve = tfp.vi.fit_surrogate_posterior(target_log_prob_fn=model.joint_log_prob(observed_time_series=pavementTemperature_train),
    surrogate_posterior=variational_posteriors,
    optimizer=optimizer,
    num_steps=num_variational_steps,)

elbo_loss_curve=np.array(elbo_loss_curve)

q_samples_demand_ = variational_posteriors.sample(50)


#进行预测
demand_forecast_dist = tfp.sts.forecast(model=model,observed_time_series=pavementTemperature_train,parameter_samples=q_samples_demand_,num_steps_forecast=24)


num_samples=30

(
    demand_forecast_mean,
    demand_forecast_scale,
    demand_forecast_samples
) = (
    demand_forecast_dist.mean().numpy()[..., 0],
    demand_forecast_dist.stddev().numpy()[..., 0],
    demand_forecast_dist.sample(num_samples).numpy()[..., 0]
    )

fig, ax = plot_forecast(date, pavementTemperature,
                           demand_forecast_mean,
                           demand_forecast_scale,
                           demand_forecast_samples,
                           title="Pavement Temperature forecast")
