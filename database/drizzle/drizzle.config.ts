/**
 * Drizzle ORM Configuration
 *
 * Configuration for database migrations and schema management.
 *
 * Environment Variables Required:
 *   DATABASE_URL - PostgreSQL connection string (Supabase)
 *
 * Usage:
 *   npx drizzle-kit generate:pg  # Generate migrations
 *   npx drizzle-kit push:pg      # Push schema to database
 *   npx drizzle-kit studio       # Open Drizzle Studio
 */

import type { Config } from 'drizzle-kit';
import * as dotenv from 'dotenv';

// Load environment variables
dotenv.config({ path: '../../.env' });

export default {
  schema: './schema.ts',
  out: './migrations',
  driver: 'pg',
  dbCredentials: {
    connectionString: process.env.DATABASE_URL || '',
  },
  verbose: true,
  strict: true,
  // Specify schemas to include
  tablesFilter: ['intelligence.*', 'execution.*', 'simulation.*', 'derivatives.*'],
} satisfies Config;
