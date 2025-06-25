import typescript from 'rollup-plugin-typescript2';
import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import { terser } from 'rollup-plugin-terser';
import pkg from './package.json';

const isProduction = process.env.NODE_ENV === 'production';

const external = [
  ...Object.keys(pkg.dependencies || {}),
  ...Object.keys(pkg.peerDependencies || {}),
  'ws',
  'node-fetch',
  'eventemitter3',
  'cross-fetch',
  'uuid',
  'retry',
  'form-data'
];

const plugins = [
  resolve({
    preferBuiltins: true,
    browser: false
  }),
  commonjs({
    include: /node_modules/
  }),
  typescript({
    typescript: require('typescript'),
    tsconfig: 'tsconfig.json',
    clean: true,
    exclude: [
      '**/*.test.ts',
      '**/*.spec.ts',
      'tests/**/*'
    ]
  })
];

if (isProduction) {
  plugins.push(
    terser({
      format: {
        comments: false
      },
      mangle: {
        keep_classnames: true,
        keep_fnames: true
      }
    })
  );
}

export default [
  // CommonJS build
  {
    input: 'src/index.ts',
    output: {
      file: pkg.main,
      format: 'cjs',
      sourcemap: true,
      exports: 'named'
    },
    external,
    plugins
  },
  
  // ES Module build
  {
    input: 'src/index.ts',
    output: {
      file: pkg.module,
      format: 'es',
      sourcemap: true
    },
    external,
    plugins
  },
  
  // Browser UMD build
  {
    input: 'src/index.ts',
    output: {
      file: 'dist/index.umd.js',
      format: 'umd',
      name: 'PRSMSdk',
      sourcemap: true,
      globals: {
        'ws': 'WebSocket',
        'node-fetch': 'fetch',
        'eventemitter3': 'EventEmitter',
        'cross-fetch': 'fetch'
      }
    },
    plugins: [
      resolve({
        preferBuiltins: false,
        browser: true
      }),
      commonjs({
        include: /node_modules/
      }),
      typescript({
        typescript: require('typescript'),
        tsconfig: 'tsconfig.json',
        clean: true,
        exclude: [
          '**/*.test.ts',
          '**/*.spec.ts',
          'tests/**/*'
        ]
      }),
      ...(isProduction ? [terser({
        format: {
          comments: false
        }
      })] : [])
    ]
  }
];