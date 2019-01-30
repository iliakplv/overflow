na = 'N/A'

target = 'CareerSatisfaction'

feature_defaults = {'Hobby': 'No',
                    'OpenSource': 'No',
                    'Student': 'No',
                    'Employment': 'Employed full-time',
                    'FormalEducation': 'Bachelor’s degree (BA, BS, B.Eng., etc.)',
                    'UndergradMajor': na,
                    'CompanySize': na,
                    'YearsCoding': '3-5 years',
                    'YearsCodingProf': '3-5 years',
                    'JobSearchStatus': 'I’m not actively looking, but I am open to new opportunities',
                    'LastNewJob': 'Less than a year ago',
                    'WakeTime': na,
                    'HoursComputer': na,
                    'HoursOutside': na,
                    'SkipMeals': na,
                    'Exercise': na,
                    'Age': na,
                    'Dependents': na}

feature_names = list(feature_defaults.keys())

record_defaults = [feature_defaults[name] for name in feature_names]
