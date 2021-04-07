from matplotlib import pylab as plt
import matplotlib.dates as mdates
import seaborn as sns
import collections
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from pandas.plotting import register_matplotlib_converters
from tensorflow_probability import sts

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

###设置图像格式
tf.enable_v2_behavior()
register_matplotlib_converters()
sns.set_context("notebook", font_scale=1.)
sns.set_style("whitegrid")
###初始化数据文件
###初始化数据文件
demand_loc = mdates.WeekdayLocator(byweekday=mdates.WE)
demand_fmt = mdates.DateFormatter('%a %b %d')

data=pd.read_excel('2013-12-16.15_2014-1-7.1.xlsx')
date = np.arange('2013-12-26T15:00', '2014-01-07T03:00', dtype='datetime64[h]')
# date=data['时间']
# date=np.array(date)

airTemperature=data['大气温度']
pavementTemperature=data['路表温度']
pavementTemperature=np.array(pavementTemperature)


num_forecast_steps = 24 # Two hour
pavementTemperature_training_data = pavementTemperature[:-num_forecast_steps]

###开始绘制初始图形
plot_compare_variables(date,
                          airTemperature,
                          pavementTemperature,
                          'airTemperature','pavementTemperature','shown the two variable',
                          num_forecast_steps)

##搭建时间序列模型
def build_model(observed_time_series):
    hour_of_day_effect = sts.Seasonal(
        num_seasons=24,
        observed_time_series=observed_time_series,
        name='hour_of_day_effect')
    temperature_effect = sts.LinearRegression(
      design_matrix=tf.reshape(airTemperature - np.mean(airTemperature),
                               (-1, 1)), name='temperature_effect')
    autoregressive = sts.Autoregressive(
      order=1,
      observed_time_series=observed_time_series,
      name='autoregressive')

    model = sts.Sum([
                  hour_of_day_effect,
                   temperature_effect,
                   autoregressive],
                   observed_time_series=observed_time_series)
    return model

demand_model = build_model(pavementTemperature_training_data)





variational_posteriors = tfp.sts.build_factored_surrogate_posterior(model=demand_model)

num_variational_steps = 400
num_variational_steps = int(num_variational_steps)

optimizer = tf.optimizers.Adam(learning_rate=0.1)
# Using fit_surrogate_posterior to build and optimize the variational loss function.
@tf.function(experimental_compile=True)
def train():
  elbo_loss_curve = tfp.vi.fit_surrogate_posterior(
    target_log_prob_fn=demand_model.joint_log_prob(
        observed_time_series=pavementTemperature_training_data),
    surrogate_posterior=variational_posteriors,
    optimizer=optimizer,
    num_steps=num_variational_steps)
  return elbo_loss_curve

elbo_loss_curve = train()
elbo_loss_curve=np.array(elbo_loss_curve)
import dataProcessFiction as dp
dp.flieWrite(elbo_loss_curve,'/Users/liyueyan/Desktop/loss.xlsx')
plt.plot(elbo_loss_curve)
plt.show()

#从训练文件中提取参数
q_samples_demand_ = variational_posteriors.sample(50)

##展示参数
print("Inferred parameters:")
for param in demand_model.parameters:
  print("{}: {} +- {}".format(param.name,
                              np.mean(q_samples_demand_[param.name], axis=0),
                              np.std(q_samples_demand_[param.name], axis=0)))


#进行预测
demand_forecast_dist = tfp.sts.forecast(
    model=demand_model,
    observed_time_series=pavementTemperature_training_data,
    parameter_samples=q_samples_demand_,
    num_steps_forecast=num_forecast_steps)

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

##查看训练数据的分布
component_dists = sts.decompose_by_component(
    demand_model,
    observed_time_series=pavementTemperature_training_data,
    parameter_samples=q_samples_demand_)

forecast_component_dists = sts.decompose_forecast_by_component(
    demand_model,
    forecast_dist=demand_forecast_dist,
    parameter_samples=q_samples_demand_)

demand_component_means_, demand_component_stddevs_ = (
    {k.name: c.mean() for k, c in component_dists.items()},
    {k.name: c.stddev() for k, c in component_dists.items()})

(
    demand_forecast_component_means_,
    demand_forecast_component_stddevs_
) = (
    {k.name: c.mean() for k, c in forecast_component_dists.items()},
    {k.name: c.stddev() for k, c in forecast_component_dists.items()}
    )

####可视化显示预测数据和画图融合的关系
component_with_forecast_means_ = collections.OrderedDict()
component_with_forecast_stddevs_ = collections.OrderedDict()
for k in demand_component_means_.keys():
  component_with_forecast_means_[k] = np.concatenate([
      demand_component_means_[k],
      demand_forecast_component_means_[k]], axis=-1)
  component_with_forecast_stddevs_[k] = np.concatenate([
      demand_component_stddevs_[k],
      demand_forecast_component_stddevs_[k]], axis=-1)

##画图展示
fig, axes = Ft.plot_components(
  date,
  component_with_forecast_means_,
  component_with_forecast_stddevs_,
  x_locator=demand_loc, x_formatter=demand_fmt)
for ax in axes.values():
  ax.axvline(date[-num_forecast_steps], linestyle="--", color='red')

##检测异常
demand_one_step_dist = sts.one_step_predictive(
    demand_model,
    observed_time_series=pavementTemperature,
    parameter_samples=q_samples_demand_)

demand_one_step_mean, demand_one_step_scale = (
    demand_one_step_dist.mean().numpy(), demand_one_step_dist.stddev().numpy())

##可视化检测的异常情况
fig, ax = plot_one_step_predictive(
    date, pavementTemperature,
    demand_one_step_mean, demand_one_step_scale,
    x_locator=demand_loc, x_formatter=demand_fmt)

# 预测时间
zscores = np.abs((pavementTemperature - demand_one_step_mean) /
                 demand_one_step_scale)
anomalies = zscores > 3.0
ax.scatter(date[anomalies],
           pavementTemperature[anomalies],
           c="red", marker="x", s=20, linewidth=2, label=r"Anomalies (>3$\sigma$)")
ax.plot(date, zscores, color="black", alpha=0.1, label='predictive z-score')
ax.legend()
plt.show()